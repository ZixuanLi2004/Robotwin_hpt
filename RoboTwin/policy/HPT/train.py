import hydra
from omegaconf import OmegaConf
import pathlib
import torch
from torch.utils.data import DataLoader
import wandb
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPT')))
from hpt.dataset.local_traj_dataset import LocalTrajDataset
from process_data import convert_dataset_robotwin_cached
from hpt.models.policy import Policy
from hpt.utils import utils
from hpt.utils.warmup_lr_wrapper import WarmupLR
from tqdm import trange
from hpt.train_test import train
from hpt.train_test import test


MAX_EPOCHS = 1000
TEST_FREQ = 10
MODEL_SAVING_FREQ = 10
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_dataset(cfg):
    dataset_args = OmegaConf.to_container(cfg.dataset, resolve=True)
    def env_rollout_fn():
        return convert_dataset_robotwin_cached(
            cfg.task_name, cfg.task_config, cfg.episode_num, use_cache=cfg.get("use_cache", True)
        )
    dataset_args["env_rollout_fn"] = env_rollout_fn
    return LocalTrajDataset(**dataset_args)

def init_policy(cfg, dataset, domain, device):
    pretrained_exists = len(cfg.train.pretrained_dir) > len("output/") and os.path.exists(
        os.path.join(cfg.train.pretrained_dir, f"trunk.pth")
    )

    if pretrained_exists:
        print("load pretrained trunk config")
        pretrained_cfg = OmegaConf.load(cfg.train.pretrained_dir + "/config.yaml")
        pretrained_cfg = OmegaConf.structured(pretrained_cfg)
        pretrained_cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(pretrained_cfg.network).to(device)
        print("load trunk from local disk")
    elif "hf" in cfg.train.pretrained_dir:
        policy = Policy.from_pretrained(cfg.train.pretrained_dir)
        print("load trunk from cloud")
    else:
        policy = hydra.utils.instantiate(cfg.network).to(device)
        print("train from scratch!!!")

    utils.update_network_dim(cfg, dataset, policy)
    policy.init_domain_stem(domain, cfg.stem)
    normalizer = dataset.get_normalizer()
    policy.init_domain_head(domain, normalizer, cfg.head)

    if cfg.network.finetune_encoder:
        utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), cfg.dataset.image_encoder)
        from hpt.utils.utils import global_language_model, global_vision_model
        policy.init_encoders("image", global_vision_model)
    
    policy.finalize_modules()
    if pretrained_exists:
        policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"trunk.pth"))

    if cfg.train.freeze_trunk:
        policy.freeze_trunk()
        print("trunk frozen")

    policy.print_model_stats()
    policy.to(device)
    return policy

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("HPT/experiments/configs")),
    config_name="config"
)
def main(cfg: OmegaConf):
    # 1. Initialize wandb
    date = cfg.output_dir.split("/")[1] if "/" in cfg.output_dir else "unknown_date"
    run = wandb.init(
        project="hpt-transfer",
        tags=[cfg.get("wb_tag", "default")],
        name=f"{date}_{cfg.get('script_name', 'hpt_train')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=False,
        save_code=False,
        resume="allow",
    )
    utils.set_seed(cfg.seed)
    print(f"train policy models with pretrained: {cfg.train.pretrained_dir}!")
    print("wandb url:", wandb.run.get_url())

    # 2. Load dataset
    dataset = get_dataset(cfg)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    val_dataset = dataset.get_validation_dataset()
    test_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=cfg.val_dataloader.shuffle,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=cfg.val_dataloader.pin_memory,
    )

    # 3. Initialize model
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0]
    
    # Define device
    gpu_id = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    policy = init_policy(cfg, dataset, domain, device)

    # 4. Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    scheduler = utils.get_scheduler(cfg.lr_scheduler, optimizer=optimizer)
    scheduler = WarmupLR(scheduler, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

    # 5. Training main loop
    utils.save_args_hydra(cfg.output_dir, cfg)
    epoch_size = len(train_loader)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    cfg.total_num_traj = dataset.replay_buffer.n_episodes
    policy_path = os.path.join(cfg.output_dir, "model.pth")
    print(f"Epoch size: {epoch_size} Traj: {cfg.total_num_traj} Train: {len(dataset)} Test: {len(val_dataset)}")

    pbar = trange(MAX_EPOCHS, position=0)
    for epoch in pbar:
        # Train step
        train_stats = train(cfg.log_interval, policy, device, train_loader, optimizer, scheduler, epoch)
        train_steps = (epoch + 1) * len(train_loader)
        
        # Test step
        if epoch % TEST_FREQ == 0:
            test_loss = test(policy, device, test_loader, epoch)
            wandb.log({"validate/epoch": epoch, f"validate/{domain}_test_loss": test_loss})
            
        if "loss" in train_stats:
            print(f"Steps: {train_steps}. Train loss: {train_stats['loss']:.5f}. Test loss: {test_loss:.5f}")

        # Save model
        if epoch % MODEL_SAVING_FREQ == 0:
            policy.save(policy_path)
        
        # Check if we should stop early
        if train_steps > cfg.train.total_iters:
            break

    # 6. Save final model and finish
    policy.save(policy_path)
    print("model saved to :", policy_path)
    utils.save_args_hydra(cfg.output_dir, cfg)
    pbar.close()
    run.finish()
    wandb.finish()

if __name__ == "__main__":
    main()