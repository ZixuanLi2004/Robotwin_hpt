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
from hpt.train_test import test 

MAX_EPOCHS = 1000
TEST_FREQ = 10
MODEL_SAVING_FREQ = 200
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_dataset(cfg):
    dataset_args = OmegaConf.to_container(cfg.dataset, resolve=True)
    def env_rollout_fn():
        return convert_dataset_robotwin_cached(
            cfg.task_name, cfg.task_config, cfg.episode_num, use_cache=cfg.get("use_cache", True)
        )
    dataset_args["env_rollout_fn"] = env_rollout_fn()
    return LocalTrajDataset(**dataset_args)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("HPT/experiments/configs")),
    config_name="config"
)
def main(cfg: OmegaConf):
    # 1. Initialize wandb
    run = wandb.init(
        project="hpt-transfer",
        tags=[cfg.get("wb_tag", "default")],
        name=f"{cfg.get('script_name', 'hpt_train')}",
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
    normalizer = dataset.get_normalizer()

    # 3. Initialize model
    domain_name = cfg.get("domains", "Robotwin")
    pretrained_exists = len(cfg.train.pretrained_dir) > len("output/") and os.path.exists(
        os.path.join(cfg.train.pretrained_dir, f"trunk.pth")
    )

    # Define device
    gpu_id = 6
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if pretrained_exists:
        print("load pretrained trunk config")
        pretrained_cfg = OmegaConf.load(cfg.train.pretrained_dir + "/config.yaml")
        pretrained_cfg = OmegaConf.structured(pretrained_cfg)
        pretrained_cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(pretrained_cfg.network).to(device)
        print("load trunk from local disk")
    elif "hf" in cfg.train.pretrained_dir:
        policy = Policy.from_pretrained(cfg.train.pretrained_dir)
        policy.to(device)  # Ensure the model is on the correct device
        print("load trunk from cloud")
    else:
        policy = hydra.utils.instantiate(cfg.network).to(device)
        print("train from scratch!!!")

    utils.update_network_dim(cfg, dataset, policy)
    policy.init_domain_stem(domain_name, cfg.stem)
    policy.init_domain_head(domain_name, normalizer, cfg.head)
    policy.finalize_modules()

    if pretrained_exists:
        policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"trunk.pth"))
        policy.to(device)  # Ensure the model is on the correct device after loading

    if cfg.train.freeze_trunk:
        for param in policy.trunk.parameters():
            param.requires_grad = False
        print("trunk frozen")

    policy.print_model_stats()

    # 4. Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    scheduler = utils.get_scheduler(cfg.lr_scheduler, optimizer=optimizer)
    scheduler = WarmupLR(scheduler, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

    # 5. Training main loop
    epoch_size = len(train_loader)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    cfg.total_num_traj = dataset.replay_buffer.n_episodes
    policy_path = os.path.join(cfg.output_dir, "model.pth")
    print(f"Epoch size: {epoch_size} Traj: {cfg.total_num_traj} Train: {len(dataset)} Test: {len(val_dataset)}")

    pbar = trange(MAX_EPOCHS, position=0)
    for epoch in pbar:
        for batch in train_loader:
            # Ensure all tensors are on the same device
            batch["data"] = {k: v.to(device).float() for k, v in batch["data"].items()}
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        wandb.log({"train/loss": loss.item(), "epoch": epoch})
        print(f"Epoch {epoch} finished. Loss: {loss.item():.4f}")

        if epoch % TEST_FREQ == 0:
            test_loss = test(policy, device, test_loader, epoch)  # Use the imported test function
            wandb.log({"validate/epoch": epoch, f"validate/{domain_name}_test_loss": test_loss})
            print(f"Epoch {epoch}. Test loss: {test_loss:.4f}")

        if (epoch + 1) % MODEL_SAVING_FREQ == 0:
            torch.save(policy.state_dict(), policy_path)
            print(f"Model saved to {policy_path}")

        # if (epoch + 1) * len(train_loader) > cfg.train.total_iters:
        #     break

    # 6. Save model
    torch.save(policy.state_dict(), policy_path)
    print("Training completed. Model saved.")

    run.finish()

if __name__ == "__main__":
    main()