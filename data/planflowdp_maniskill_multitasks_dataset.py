import torch
import numpy as np
import sys
import json
#sys.path.append('/home/lasserre/changhe/MoTVLA')
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import h5py
import os
import pickle
import copy
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class MultiTaskManiSkillVLMDataset(torch.utils.data.Dataset):
    def __init__(self,
                 action_data_dir: str,  # Directory containing action .pkl files
                 language_data_dir: str,  # Directory containing language .npz files
                 pred_horizon: int = 20,
                 obs_horizon: int = 5,
                 action_horizon: int = 15,
                 n_tokens: int = 16,
                 num_demos_per_task: int = -1,  # -1 means use all demos
                 resize_scale: int = 96,
                 pretrained: bool = False,
                 tasks: Optional[List[str]] = None,  # Specific tasks to load
                 normalize_across_tasks: bool = True,
                 #balance_tasks: bool = True,
                 file_pattern: str = "*.pkl",  # Pattern for action files
                 language_file_pattern: str = "*.jsonl"):  # Pattern for language files
        
        self.action_data_dir = Path(action_data_dir)
        self.language_data_dir = Path(language_data_dir)
        
        # Discover available tasks if not specified
        if tasks is None:
            # Find all action files and extract task names
            action_files = list(self.action_data_dir.glob(file_pattern))
            # Extract task names from filenames
            # Assumes format like: maniskill_[taskname]_dp_action_[demos].pkl
            tasks = []
            for f in action_files:
                # Try to extract task name from filename
                filename = f.stem  # Remove .pkl extension
                # Simple heuristic: extract part between 'maniskill_' and '_dp_action' or similar
                parts = filename.split('_')
                if 'maniskill' in parts[0].lower():
                    # Find the task name parts (between maniskill and dp/action)
                    task_parts = []
                    for i, part in enumerate(parts[1:], 1):
                        if part in ['dp', 'action', 'language']:
                            break
                        task_parts.append(part)
                    if task_parts:
                        task_name = '_'.join(task_parts)
                        tasks.append(task_name)
                else:
                    # Use the whole filename as task name if pattern doesn't match
                    tasks.append(filename)
        
        print(f"Loading data for tasks: {tasks}")
        
        # Store parameters
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.n_tokens = n_tokens
        self.resize_scale = resize_scale
        self.pretrained = pretrained
        self.normalize_across_tasks = normalize_across_tasks
        #self.balance_tasks = balance_tasks
        self.num_demos_per_task = num_demos_per_task
        
        # Initialize storage
        self.task_names = []
        self.task_data = []
        self.task_indices = []
        self.task_stats = []
        self.all_indices_with_task = []
        self.task_episode_ends = []
        self.task_cumulative_frames = []
        
        # Collect all data for global statistics if needed
        all_actions = []
        all_agent_poses = []
        cumulative_frames = 0
        
        # Load data for each task
        for task_idx, task in enumerate(tasks):
            # Find action data file
            action_files = list(self.action_data_dir.glob(f"*{task}*.pkl"))
            if not action_files:
                print(f"Warning: No action file found for task {task}, skipping")
                continue
            action_file = action_files[0]  # Take first matching file
            
            # Find language data file
            language_files = list(self.language_data_dir.glob(f"*{task}*.jsonl")) 
            
            if not language_files:
                print(f"Warning: No language file found for task {task}, skipping")
                continue
            language_file = language_files[0]  # Take first matching file
            
            print(f"Task {task}:")
            print(f"  Action file: {action_file.name}")
            print(f"  Language file: {language_file.name}")
            
            # Load action dataset
            if action_file.suffix == '.npy':
                dataset_root = np.load(str(action_file), allow_pickle=True)
                data_dict = dataset_root.item()
            else:  # .pkl
                with open(str(action_file), 'rb') as f:
                    data_dict = pickle.load(f)
            
            # Load language sentences from JSONL
            sentences = []
            with open(str(language_file), 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    gpt_value = next(
                        conv['value'] for conv in entry['conversations']
                        if conv['from'] == 'gpt'
                    )
                    sentences.append(gpt_value)
            language_sentences = np.array(sentences, dtype=object)
            
            # Extract data
            end_frames = data_dict['end_frames']
            actions = data_dict['all_actions']
            agent_pose = data_dict['all_agent_pose']
            images_camera0 = data_dict['all_images_camera0']
            
            # (language sentences already loaded above as language_sentences)
            
            # Determine number of demos to use
            num_max_demos = end_frames.shape[0]
            if self.num_demos_per_task > 0:
                num_demos = min(num_max_demos, self.num_demos_per_task)
            else:
                num_demos = num_max_demos
            
            num_max_frames = end_frames[num_demos - 1]
            episode_ends = end_frames[:num_demos]
            
            print(f"  Episodes: {len(episode_ends)}, Frames: {num_max_frames}")
            
            # Create train data for this task
            train_data = {
                'images_camera0': images_camera0[:num_max_frames],
                'agent_pos': agent_pose[:num_max_frames],
                'action': actions[:num_max_frames],
                'language': language_sentences[:num_max_frames],
            }
            
            # Collect for global statistics if needed
            if self.normalize_across_tasks:
                all_actions.append(actions[:num_max_frames])
                all_agent_poses.append(agent_pose[:num_max_frames])
            
            # Compute indices for this task
            indices = create_sample_indices(
                episode_ends=episode_ends,
                sequence_length=pred_horizon,
                pad_before=obs_horizon - 1,
                pad_after=action_horizon - 1
            )
            
            # Store task-specific data
            self.task_names.append(task)
            self.task_data.append(train_data)
            self.task_indices.append(indices)
            self.task_episode_ends.append(episode_ends)
            self.task_cumulative_frames.append(cumulative_frames)
            
            # Add to combined indices list with task identifier
            for idx_tuple in indices:
                self.all_indices_with_task.append((task_idx, idx_tuple))
            
            cumulative_frames += num_max_frames
            
            print(f"  Loaded {len(indices)} samples from {num_demos} demos")
        
        if len(self.task_names) == 0:
            raise ValueError("No valid tasks found in the specified directories")
        
        self.num_tasks = len(self.task_names)
        
        # Compute normalization statistics
        if self.normalize_across_tasks and len(all_actions) > 0:
            # Compute global statistics across all tasks
            print("\nComputing global normalization statistics...")
            all_actions = np.concatenate(all_actions, axis=0)
            all_agent_poses = np.concatenate(all_agent_poses, axis=0)
            
            self.global_stats = {
                'action': get_data_stats(all_actions),
                'agent_pos': get_data_stats(all_agent_poses)
            }
            
            self.stats = self.global_stats

            # Normalize data for each task using global statistics
            for task_id, train_data in enumerate(self.task_data):
                normalized_data = {}
                for key, data in train_data.items():
                    if key in ['action', 'agent_pos']:
                        normalized_data[key] = normalize_data(data, self.global_stats[key])
                    else:
                        normalized_data[key] = data
                self.task_data[task_id] = normalized_data
        else:
            # Compute and apply per-task statistics
            print("\nComputing per-task normalization statistics...")
            for task_id, train_data in enumerate(self.task_data):
                task_stats = {}
                normalized_data = {}
                
                for key, data in train_data.items():
                    if key in ['action', 'agent_pos']:
                        task_stats[key] = get_data_stats(data)
                        normalized_data[key] = normalize_data(data, task_stats[key])
                    else:
                        normalized_data[key] = data
                
                self.task_data[task_id] = normalized_data
                self.task_stats.append(task_stats)
        
        # Setup image transforms
        if self.pretrained:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.resize_scale, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.resize_scale, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
        
        # # Create balanced sampling indices if needed
        # if self.balance_tasks:
        #     self._create_balanced_indices()
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Dataset initialized with {self.num_tasks} tasks")
        print(f"Total samples: {len(self)}")
        print(f"{'='*50}")
        for task_id, task_name in enumerate(self.task_names):
            task_samples = sum(1 for tid, _ in self.all_indices_with_task if tid == task_id)
            print(f"  {task_name:20s}: {task_samples:6d} samples")
        print(f"{'='*50}")
    
    # def _create_balanced_indices(self):
    #     """Create balanced indices for equal sampling across tasks"""
    #     # Group indices by task
    #     task_grouped_indices = [[] for _ in range(self.num_tasks)]
    #     for idx, (task_id, sample_indices) in enumerate(self.all_indices_with_task):
    #         task_grouped_indices[task_id].append(idx)
        
    #     # Find max samples per task
    #     max_samples = max(len(indices) for indices in task_grouped_indices)
        
    #     # Create balanced indices by repeating samples from smaller tasks
    #     self.balanced_indices = []
    #     for task_indices in task_grouped_indices:
    #         if len(task_indices) == 0:
    #             continue
    #         # Repeat indices to match max_samples
    #         repeated_indices = []
    #         while len(repeated_indices) < max_samples:
    #             repeated_indices.extend(task_indices)
    #         self.balanced_indices.extend(repeated_indices[:max_samples])
    
    def __len__(self):
        # if self.balance_tasks and hasattr(self, 'balanced_indices'):
        #     return len(self.balanced_indices)
        return len(self.all_indices_with_task)
    
    def __getitem__(self, idx):
        # Get the actual index based on balancing
        # if self.balance_tasks and hasattr(self, 'balanced_indices'):
        #     idx = self.balanced_indices[idx]
        
        # Get task ID and sample indices
        task_id, sample_indices = self.all_indices_with_task[idx]
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sample_indices
        
        # Get the normalized data for this task
        normalized_train_data = self.task_data[task_id]
        
        # Sample sequence
        nsample = sample_sequence(
            train_data=normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        
        # Apply image transforms
        image_key = 'images_camera0'
        if image_key in nsample:
            images = nsample[image_key][:self.obs_horizon]
            transformed_images = []
            for img in images:
                transformed_img = self.transform(img)
                transformed_images.append(transformed_img)
            nsample[image_key] = torch.stack(transformed_images)
        
        # Slice observations to obs_horizon
        nsample['images_camera0'] = nsample['images_camera0'][:self.obs_horizon]
        nsample['language'] = [str(x) for x in nsample['language'][:self.obs_horizon]]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        
        # Add task information
        nsample['task_id'] = task_id
        nsample['task_name'] = self.task_names[task_id]
        
        return nsample
    
    def get_normalizer(self):
        """Get normalizer for the dataset (useful for inference)"""
        if self.normalize_across_tasks:
            return self.global_stats
        else:
            return self.task_stats
    
    def get_task_info(self):
        """Get information about loaded tasks"""
        info = {
            'task_names': self.task_names,
            'num_tasks': self.num_tasks,
            'task_episode_counts': [len(ep_ends) for ep_ends in self.task_episode_ends],
            'task_frame_counts': [ep_ends[-1] for ep_ends in self.task_episode_ends],
            'total_samples': len(self.all_indices_with_task)
        }
        return info


if __name__ == "__main__":
    # Example usage with folder paths (automatically discovers tasks)
    dataset = MultiTaskManiSkillVLMDataset(
        action_data_dir='/home/aiden/MoT_maniskill_sort_100/action_data/',
        language_data_dir='/home/aiden/MoT_maniskill_sort_100/language_data/',
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_demos_per_task=3,  # Use 50 demos per task, -1 for all
        resize_scale=96,
        pretrained=False,
        normalize_across_tasks=True,
        #balance_tasks=True,
        # Optional: specify specific tasks
        #tasks=['two_step_pulltool']
    )
    
    # Get task information
    task_info = dataset.get_task_info()
    print(f"\nTask Information:")
    for key, value in task_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test sampling from different tasks
    for i in range(min(3, len(dataset))):
        sample = dataset[i * len(dataset) // 3]
        print(f"\nSample {i}:")
        print(f"  Task: {sample['task_name']} (ID: {sample['task_id']})")
        print(f"  Agent pos shape: {sample['agent_pos'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Camera0 images shape: {sample['images_camera0'].shape}")
        print(sample["language"])

        # images = sample["images_camera0"]   # [obs_horizon, C, H, W]
        # texts = sample["language"]

        # for t in range(images.shape[0]):
        #     img_path = f"sample{i}_frame{t}.png"
        #     from torchvision.utils import save_image
        #     save_image(images[t], img_path)
        #     print(f"  Saved image to: {img_path}")
        #     print(f"  Corresponding text: {texts[t]}")