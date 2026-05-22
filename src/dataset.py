import torch
import json
import os
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader, cpu
import numpy as np

class SurgicalVideoDataset(Dataset):
    def __init__(self, jsonl_path, processor, video_folder, num_frames=8, max_length=1024):
        # ❌ 删除了这行：self.data = [json.loads(line) for line in open(jsonl_path...)]
        self.processor = processor
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.max_length = max_length
        self.examples = []
        
        # 🌟 核心拦截逻辑：读取 jsonl 时进行物理验活
        missing_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                video_path = os.path.join(self.video_folder, data["video"])
                
                # 只有当硬盘里真的有这个 MP4 文件时，才加入训练集 (唯一数据源)
                if os.path.exists(video_path):
                    self.examples.append(data)
                else:
                    missing_count += 1
                    
        print(f" 数据集加载完毕！有效样本数: {len(self.examples)}")
        if missing_count > 0:
            print(f" 已自动清理 {missing_count} 个因边界错位缺失的尾部幽灵片段。")

    def _load_video(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(f) for f in frames]
        except Exception as e:
            print(f"\n[Error] Failed to load video {video_path}: {e}")
            # 返回黑屏帧作为 Fallback，防止训练直接崩溃
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return [Image.fromarray(dummy_frame) for _ in range(self.num_frames)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 🌟 修改点：从过滤后的干净列表里取数据
        item = self.examples[idx] 
        video_file = os.path.join(self.video_folder, item["video"])
        
        pixel_values = self._load_video(video_file)
        
        # 统一使用全英文 Prompt，对齐之前的 Route B 数据集
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pixel_values},
                    {"type": "text", "text": "Please observe this short clip of cataract surgery and generate a detailed surgical report."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["conversations"][1]["value"]}
                ],
            }
        ]

        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[texts],
            videos=[pixel_values],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
    
def collate_fn(batch):
    """
    智能批处理函数：动态适配不同版本的 Qwen Processor 输出键值
    """
    # 1. 提取并对齐文本特征 (需要 Padding)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], 
        batch_first=True, 
        padding_value=0
    )

    # 2. 生成训练标签 (忽略 Padding 部分的 Loss)
    labels = input_ids.clone()
    labels[labels == 0] = -100 

    # 3. 基础字典
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    # 4. 动态提取所有视觉相关的特征 (如 pixel_values, pixel_values_videos, video_grid_thw 等)
    # 遍历当前 batch 第一个样本拥有的所有 key
    first_item_keys = batch[0].keys()
    
    for key in first_item_keys:
        if key not in ["input_ids", "attention_mask", "labels"]:
            # 将该特征在 batch 维度上堆叠 (使用 torch.stack 或是简单的列表聚合视情况而定)
            # 因为 Qwen-VL 的 pixel_values 往往是 2D 张量，直接使用 torch.cat 沿第 0 维拼接
            if isinstance(batch[0][key], torch.Tensor):
                 batch_dict[key] = torch.cat([item[key] for item in batch], dim=0)

    return batch_dict