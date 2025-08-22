import os
import numpy as np
import hydra

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPT')))
from hpt.utils import utils
from hpt import train_test
from collections import OrderedDict

def encode_obs(observation, instruction):
    obs = {}
    # 关节状态
    obs["state"] = observation["joint_action"]["vector"]
    # 相机图像处理
    if "observation" in observation:
        cams = observation["observation"]
        if "head_camera" in cams and "rgb" in cams["head_camera"]:
            img = cams["head_camera"]["rgb"]
            obs["image"] = np.moveaxis(img, -1, 0) / 255.0
        if "left_camera" in cams and "rgb" in cams["left_camera"]:
            img = cams["left_camera"]["rgb"]
            obs["image1"] = np.moveaxis(img, -1, 0) / 255.0
        if "right_camera" in cams and "rgb" in cams["right_camera"]:
            img = cams["right_camera"]["rgb"]
            obs["image2"] = np.moveaxis(img, -1, 0) / 255.0
    obs["language_instruction"] = instruction
    return OrderedDict(obs)

def get_model(usr_args):
    # 初始化 policy
    from omegaconf import OmegaConf
    cfg_path = usr_args["cfg_path"]
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    utils.set_seed(cfg.seed)
    model_dir = './policy/HPT/output/hpt_train'
    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0]

    policy = hydra.utils.instantiate(cfg.network).to(device)
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, None, cfg.head)
    policy.finalize_modules()
    policy.print_model_stats()
    utils.set_seed(cfg.seed)
    # 加载 encoder
    if cfg.network.finetune_encoder:
        utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), cfg.dataset.image_encoder)
        from hpt.utils.utils import global_vision_model
        policy.init_encoders("image", global_vision_model)
    # 加载权重
    policy.load_model(os.path.join(model_dir, "model.pth"))
    policy.to(device)
    policy.eval()
    return policy

def eval(TASK_ENV, policy, observation):
    instruction = TASK_ENV.get_instruction()
    obs = encode_obs(observation, instruction)
    action = policy.get_action(obs)
    TASK_ENV.take_action(action, action_type='qpos')

def reset_model(policy):
    if hasattr(policy, "reset"):
        policy.reset()