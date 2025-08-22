import sys
import os

sys.path.append("./policy/HPT/HPT")

import numpy as np
import argparse
import pickle
import json
from collections import OrderedDict
import h5py
import cv2


def load_hdf5(dataset_path):
    """Load data from HDF5 file in RoboTwin format"""
    if h5py is None:
        print("Error: h5py is required. Install with: pip install h5py")
        return None
        
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        return None

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def convert_dataset_robotwin(task_name, task_config, episode_num):
    """
    Generator function compatible with HPT's LocalTrajDataset
    Converts RoboTwin HDF5 data to HPT format
    """
    # 自动检测数据路径
    possible_paths = [
        os.path.join("../../data", task_name, task_config, "data"),  # 从policy/HPT运行时
        os.path.join("./data", task_name, task_config, "data"),      # 从根目录运行时
        os.path.join("data", task_name, task_config, "data"),       # 从根目录运行时(相对路径)
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Error: Could not find data directory for {task_name}/{task_config}")
        print("Tried paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        return
    
    print(f"Loading data from: {os.path.abspath(data_path)}")
    
    # Load task instruction
    instruction_file = f"../../description/task_instruction/{task_name}.json" #TODO
    language_instruction = None  # 初始化为 None
    if os.path.exists(instruction_file):
        with open(instruction_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            language_instruction = data.get("full_description", "").strip()
    
    folders = os.listdir(data_path)
    assert episode_num <= len(folders), f"Requested {episode_num} episodes but only {len(folders)} available"

    for i in range(episode_num):
        episode_path = os.path.join(data_path, f"episode{i}.hdf5")
        if not os.path.exists(episode_path):
            print(f"Warning: {episode_path} does not exist, skipping...")
            continue
            
        print(f"Processing episode {i}...")
        data = load_hdf5(episode_path)
        if data is None:
            print(f"Warning: Failed to load {episode_path}, skipping...")
            continue
            
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = data
        
        steps = []
        episode_length = left_gripper_all.shape[0]
        
        for j in range(episode_length):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            
            # Concatenate joint states (similar to other policies)
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
            state = state.astype(np.float32)
            
            # Process images
            observation = {"state": state}
            
            # Process camera images - 支持三视角
            # head_camera -> image, right_camera -> image1, left_camera -> image2
            
            if "head_camera" in image_dict:
                camera_bits = image_dict["head_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image"] = camera_resized.astype(np.float32)
            
            if "right_camera" in image_dict:
                camera_bits = image_dict["right_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image1"] = camera_resized.astype(np.float32)
            
            if "left_camera" in image_dict:
                camera_bits = image_dict["left_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image2"] = camera_resized.astype(np.float32)
            
            # Create step dict (HPT format)
            step = OrderedDict({
                "observation": observation,
                "action": state.copy(),  # Use joint state as action for next step
            })

            # Only add language_instruction if it is not empty
            if language_instruction is not None and language_instruction.strip():
                step["language_instruction"] = language_instruction
            
            steps.append(step)
        
        # Create episode dict
        episode_dict = {"steps": steps}
        yield episode_dict

def save_processed_data(episodes, task_name, task_config, episode_num, save_dir="./processed_data"):
    """
    保存处理好的数据到磁盘，供后续训练直接使用
    """
    dataset_name = f"robotwin_{task_name}_{task_config}_{episode_num}"
    full_save_dir = os.path.join(save_dir, dataset_name)
    
    # 创建保存目录
    os.makedirs(full_save_dir, exist_ok=True)
    
    print(f"Saving processed data to: {full_save_dir}")
    
    # 保存episodes数据
    episodes_file = os.path.join(full_save_dir, "episodes.pkl")
    with open(episodes_file, 'wb') as f:
        pickle.dump(episodes, f)
    
    # 保存数据集信息
    dataset_info = {
        "task_name": task_name,
        "task_config": task_config,
        "episode_num": episode_num,
        "total_episodes": len(episodes),
        "total_steps": sum(len(ep["steps"]) for ep in episodes),
        "action_dim": episodes[0]["steps"][0]["action"].shape[0] if episodes else 0,
        "state_dim": episodes[0]["steps"][0]["observation"]["state"].shape[0] if episodes else 0,
        "has_image": "image" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "has_image1": "image1" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "has_image2": "image2" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "language_instruction": episodes[0]["steps"][0].get("language_instruction", "") if episodes else "",
    }
    
    info_file = os.path.join(full_save_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    print(f"Saved {len(episodes)} episodes with {dataset_info['total_steps']} total steps")
    print(f"Dataset info saved to: {info_file}")
    
    return full_save_dir


def load_processed_data(dataset_path):
    """
    加载预处理好的数据
    """
    episodes_file = os.path.join(dataset_path, "episodes.pkl")
    info_file = os.path.join(dataset_path, "dataset_info.json")
    
    if not os.path.exists(episodes_file):
        print(f"Error: Processed data not found at {episodes_file}")
        return None, None
    
    # 加载episodes
    with open(episodes_file, 'rb') as f:
        episodes = pickle.load(f)
    
    # 加载数据集信息
    dataset_info = None
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes from {dataset_path}")
    return episodes, dataset_info


def convert_dataset_robotwin_cached(task_name, task_config, episode_num, use_cache=True, save_dir="./processed_data"):
    """
    带缓存功能的数据转换函数
    """
    dataset_name = f"robotwin_{task_name}_{task_config}_{episode_num}"
    cache_dir = os.path.join(save_dir, dataset_name)
    if use_cache and os.path.exists(cache_dir):
        print(f"Found cached data at {cache_dir}, loading...")
        episodes, dataset_info = load_processed_data(cache_dir)
        if episodes is not None:
            for episode in episodes:
                yield episode
            return
    
    # 没有缓存或不使用缓存，重新处理
    print("Processing data from scratch...")
    episodes = list(convert_dataset_robotwin(task_name, task_config, episode_num))
    
    # 保存处理好的数据
    if episodes:
        save_processed_data(episodes, task_name, task_config, episode_num, save_dir)
    
    # 生成数据
    for episode in episodes:
        yield episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RoboTwin data for HPT")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., _pick_block)",
    )
    parser.add_argument("task_config", type=str, help="Task configuration (e.g., demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="Number of expert demonstrations")
    parser.add_argument("--save_dir", type=str, default="./processed_data", help="Directory to save processed data")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data if available")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    print(f"Processing {expert_data_num} episodes for task: {task_name}, config: {task_config}")

    # 检测数据路径
    possible_paths = [
        os.path.join("../../data", task_name, task_config, "data"),  # 从policy/HPT运行时
        os.path.join("./data", task_name, task_config, "data"),      # 从根目录运行时
        os.path.join("data", task_name, task_config, "data"),       # 从根目录运行时(相对路径)
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Error: Could not find data directory for {task_name}/{task_config}")
        print("Tried paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        exit(1)

    # 默认自动保存，且支持缓存
    episodes = list(convert_dataset_robotwin_cached(
        task_name, task_config, expert_data_num, 
        use_cache=args.use_cache, save_dir=args.save_dir
    ))

    print(f"Successfully converted {len(episodes)} episodes")

    if episodes:
        first_episode = episodes[0]
        print(f"First episode has {len(first_episode['steps'])} steps")
        first_step = first_episode['steps'][0]
        print(f"Step format: observation keys = {first_step['observation'].keys()}")
        print(f"Action shape: {first_step['action'].shape}")
        language_instruction = first_step.get("language_instruction", "")
        print(f"Language: {language_instruction[:50] if language_instruction else 'No language instruction'}")

    print("HPT data processing completed successfully!")
import sys
import os

sys.path.append("./policy/HPT/HPT")

import numpy as np
import argparse
import pickle
import json
from collections import OrderedDict
import h5py
import cv2


def load_hdf5(dataset_path):
    """Load data from HDF5 file in RoboTwin format"""
    if h5py is None:
        print("Error: h5py is required. Install with: pip install h5py")
        return None
        
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        return None

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def convert_dataset_robotwin(task_name, task_config, episode_num):
    """
    Generator function compatible with HPT's LocalTrajDataset
    Converts RoboTwin HDF5 data to HPT format
    """
    # 自动检测数据路径
    possible_paths = [
        os.path.join("../../data", task_name, task_config, "data"),  # 从policy/HPT运行时
        os.path.join("./data", task_name, task_config, "data"),      # 从根目录运行时
        os.path.join("data", task_name, task_config, "data"),       # 从根目录运行时(相对路径)
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Error: Could not find data directory for {task_name}/{task_config}")
        print("Tried paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        return
    
    print(f"Loading data from: {os.path.abspath(data_path)}")
    
    # Load task instruction
    instruction_file = f"../../description/task_instruction/{task_name}.json" #TODO
    language_instruction = None  # 初始化为 None
    if os.path.exists(instruction_file):
        with open(instruction_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            language_instruction = data.get("full_description", "").strip()
    
    folders = os.listdir(data_path)
    assert episode_num <= len(folders), f"Requested {episode_num} episodes but only {len(folders)} available"

    for i in range(episode_num):
        episode_path = os.path.join(data_path, f"episode{i}.hdf5")
        if not os.path.exists(episode_path):
            print(f"Warning: {episode_path} does not exist, skipping...")
            continue
            
        print(f"Processing episode {i}...")
        data = load_hdf5(episode_path)
        if data is None:
            print(f"Warning: Failed to load {episode_path}, skipping...")
            continue
            
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = data
        
        steps = []
        episode_length = left_gripper_all.shape[0]
        
        for j in range(episode_length):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            
            # Concatenate joint states (similar to other policies)
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
            state = state.astype(np.float32)
            
            # Process images
            observation = {"state": state}
            
            # Process camera images - 支持三视角
            # head_camera -> image, right_camera -> image1, left_camera -> image2
            
            if "head_camera" in image_dict:
                camera_bits = image_dict["head_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image"] = camera_resized.astype(np.float32)

            if "left_camera" in image_dict:
                camera_bits = image_dict["left_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image1"] = camera_resized.astype(np.float32)
            
            if "right_camera" in image_dict:
                camera_bits = image_dict["right_camera"][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_resized = cv2.resize(camera_img, (224, 224))
                observation["image2"] = camera_resized.astype(np.float32)
            
            
            # Create step dict (HPT format)
            step = OrderedDict({
                "observation": observation,
                "action": state.copy(),  # Use joint state as action for next step
            })

            # Only add language_instruction if it is not empty
            if language_instruction is not None and language_instruction.strip():
                step["language_instruction"] = language_instruction
            
            steps.append(step)
        
        # Create episode dict
        episode_dict = {"steps": steps}
        yield episode_dict

def save_processed_data(episodes, task_name, task_config, episode_num, save_dir="./processed_data"):
    """
    保存处理好的数据到磁盘，供后续训练直接使用
    """
    dataset_name = f"robotwin_{task_name}_{task_config}_{episode_num}"
    full_save_dir = os.path.join(save_dir, dataset_name)
    
    # 创建保存目录
    os.makedirs(full_save_dir, exist_ok=True)
    
    print(f"Saving processed data to: {full_save_dir}")
    
    # 保存episodes数据
    episodes_file = os.path.join(full_save_dir, "episodes.pkl")
    with open(episodes_file, 'wb') as f:
        pickle.dump(episodes, f)
    
    # 保存数据集信息
    dataset_info = {
        "task_name": task_name,
        "task_config": task_config,
        "episode_num": episode_num,
        "total_episodes": len(episodes),
        "total_steps": sum(len(ep["steps"]) for ep in episodes),
        "action_dim": episodes[0]["steps"][0]["action"].shape[0] if episodes else 0,
        "state_dim": episodes[0]["steps"][0]["observation"]["state"].shape[0] if episodes else 0,
        "has_image": "image" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "has_image1": "image1" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "has_image2": "image2" in episodes[0]["steps"][0]["observation"] if episodes else False,
        "language_instruction": episodes[0]["steps"][0].get("language_instruction", "") if episodes else "",
    }
    
    info_file = os.path.join(full_save_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    print(f"Saved {len(episodes)} episodes with {dataset_info['total_steps']} total steps")
    print(f"Dataset info saved to: {info_file}")
    
    return full_save_dir


def load_processed_data(dataset_path):
    """
    加载预处理好的数据
    """
    episodes_file = os.path.join(dataset_path, "episodes.pkl")
    info_file = os.path.join(dataset_path, "dataset_info.json")
    
    if not os.path.exists(episodes_file):
        print(f"Error: Processed data not found at {episodes_file}")
        return None, None
    
    # 加载episodes
    with open(episodes_file, 'rb') as f:
        episodes = pickle.load(f)
    
    # 加载数据集信息
    dataset_info = None
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes from {dataset_path}")
    return episodes, dataset_info


def convert_dataset_robotwin_cached(task_name, task_config, episode_num, use_cache=True, save_dir="./processed_data"):
    """
    带缓存功能的数据转换函数
    """
    dataset_name = f"robotwin_{task_name}_{task_config}_{episode_num}"
    cache_dir = os.path.join(save_dir, dataset_name)
    if use_cache and os.path.exists(cache_dir):
        print(f"Found cached data at {cache_dir}, loading...")
        episodes, dataset_info = load_processed_data(cache_dir)
        if episodes is not None:
            for episode in episodes:
                yield episode
            return
    
    # 没有缓存或不使用缓存，重新处理
    print("Processing data from scratch...")
    episodes = list(convert_dataset_robotwin(task_name, task_config, episode_num))
    
    # 保存处理好的数据
    if episodes:
        save_processed_data(episodes, task_name, task_config, episode_num, save_dir)
    
    # 生成数据
    for episode in episodes:
        yield episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RoboTwin data for HPT")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., _pick_block)",
    )
    parser.add_argument("task_config", type=str, help="Task configuration (e.g., demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="Number of expert demonstrations")
    parser.add_argument("--save_dir", type=str, default="./processed_data", help="Directory to save processed data")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data if available")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    print(f"Processing {expert_data_num} episodes for task: {task_name}, config: {task_config}")

    # 检测数据路径
    possible_paths = [
        os.path.join("../../data", task_name, task_config, "data"),  # 从policy/HPT运行时
        os.path.join("./data", task_name, task_config, "data"),      # 从根目录运行时
        os.path.join("data", task_name, task_config, "data"),       # 从根目录运行时(相对路径)
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Error: Could not find data directory for {task_name}/{task_config}")
        print("Tried paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        exit(1)

    # 默认自动保存，且支持缓存
    episodes = list(convert_dataset_robotwin_cached(
        task_name, task_config, expert_data_num, 
        use_cache=args.use_cache, save_dir=args.save_dir
    ))

    print(f"Successfully converted {len(episodes)} episodes")

    if episodes:
        first_episode = episodes[0]
        print(f"First episode has {len(first_episode['steps'])} steps")
        first_step = first_episode['steps'][0]
        print(f"Step format: observation keys = {first_step['observation'].keys()}")
        print(f"Action shape: {first_step['action'].shape}")
        language_instruction = first_step.get("language_instruction", "")
        print(f"Language: {language_instruction[:50] if language_instruction else 'No language instruction'}")

    print("HPT data processing completed successfully!")
