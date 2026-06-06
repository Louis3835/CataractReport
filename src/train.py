import torch
import os
import argparse
from transformers import TrainingArguments, Trainer, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from dataset import SurgicalVideoDataset # 确保引入了我们上面修改过的 Dataset
from dataclasses import dataclass

MODEL_PATH = "/media/zack/Data/Data/models/Qwen3-VL-4B-Instruct" 
OUTPUT_DIR_BASE = "./output/qwen3_4b_vl_surg"
PROJECT_ROOT = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport"


def resolve_train_dataset_path(mode):
    data_dir = os.path.join(PROJECT_ROOT, f"data_split_{mode}")
    for filename in ("train_dataset.jsonl", "train_dataset.json"):
        candidate_path = os.path.join(data_dir, filename)
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"找不到训练集文件，已检查: {os.path.join(data_dir, 'train_dataset.jsonl')} 和 {os.path.join(data_dir, 'train_dataset.json')}"
    )


def resolve_output_dir(mode):
    return os.path.join(OUTPUT_DIR_BASE, mode)

# 🌟 新增：多模态数据碰撞器 (Data Collator)
# 作用：将不同长度的 input_ids 用 pad_token_id 补齐，组成规整的矩阵喂给 GPU
@dataclass
class MultimodalDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # 1. 补齐文本输入序列
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        # 2. 补齐 labels 序列
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": input_ids_padded.ne(self.processor.tokenizer.pad_token_id),
        }
        
        # 🌟 3. 核心修复：处理 Qwen3-VL 梦寐以求的 mm_token_type_ids
        if "mm_token_type_ids" in features[0]:
            mm_token_type_ids = [f["mm_token_type_ids"] for f in features]
            # 同样使用 pad_token_id（通常是0）对齐补齐成矩阵
            mm_token_padded = torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids, batch_first=True, padding_value=0
            )
            batch["mm_token_type_ids"] = mm_token_padded
        
        # 4. 强力对齐视频张量的 Batch 轴
        if "pixel_values_videos" in features[0]:
            batch["pixel_values_videos"] = torch.cat([f["pixel_values_videos"].view(-1, *f["pixel_values_videos"].shape[-3:]) for f in features], dim=0)
            batch["video_grid_thw"] = torch.cat([f["video_grid_thw"].view(-1, 3) for f in features], dim=0)
            
        return batch


def train(mode):
    output_dir = resolve_output_dir(mode)
    # 生成 output_dir 目录，避免 Trainer 报错
    os.makedirs(output_dir, exist_ok=True)
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 🚨 注意：目前 Huggingface 的类名通常是 Qwen2VLForConditionalGeneration (向下兼容)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
        # 🚨 警告：在使用 torchrun 或 accelerate 进行多卡分布式训练 (DDP) 时，
        # 绝不能加 device_map="auto"。Trainer 会自动负责把模型放置到对应的卡上！
    )
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    train_dataset_path = resolve_train_dataset_path(mode)
    train_dataset = SurgicalVideoDataset(train_dataset_path, processor, PROJECT_ROOT)
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # 显存不足就保持1
        gradient_accumulation_steps=8, 
        learning_rate=3e-5,            
        lr_scheduler_type="cosine",    
        warmup_ratio=0.1,              
        num_train_epochs=3,
        bf16=True,
        save_strategy="steps",
        save_steps=200,
        optim="paged_adamw_8bit",   
        weight_decay=0.01,          
        logging_steps=10,
        gradient_checkpointing=True,
        remove_unused_columns=False, # 🌟 核心开关：多模态微调必须设为 False，否则丢掉视频特征
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset,
        data_collator=MultimodalDataCollator(processor) # 🌟 注入 Collator
    )
    
    trainer.train()
    final_weights_dir = os.path.join(output_dir, "final_lora_weights")
    trainer.model.save_pretrained(final_weights_dir)
    print(f"✅ 训练完成！LoRA 权重已保存至 {final_weights_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="白内障手术多模态微调训练")
    parser.add_argument("--mode", type=str, choices=["single", "contextual"], required=True,
                        help="选择要读取的数据拆分目录: 'single' 或 'contextual'")
    args = parser.parse_args()

    train(args.mode)