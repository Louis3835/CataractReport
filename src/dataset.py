import torch
import json
import os
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

class SurgicalVideoDataset(Dataset):
    def __init__(self, jsonl_path, processor, video_folder, max_length=1024):
        self.processor = processor
        self.video_folder = video_folder
        self.max_length = max_length
        self.examples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if os.path.exists(os.path.join(self.video_folder, data["video"])):
                    self.examples.append(data)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        video_path = os.path.join(self.video_folder, item["video"])
        
        # 构建符合 Qwen3-VL 的对话模板
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path}, # 直接传路径，库会自动处理
                    {"type": "text", "text": "Observe this surgery clip and generate a report."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["conversations"][1]["value"]}],
            }
        ]
        
        # 生成 prompt 和处理视觉特征
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
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