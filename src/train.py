import torch
import os
from transformers import TrainingArguments, Trainer, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from dataset import SurgicalVideoDataset

MODEL_PATH = "/media/zack/Data/Data/models/Qwen3-VL-2B-Instruct"

def train():
    # 1. 加载原生多模态模型
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        #device_map="auto",
        trust_remote_code=True
    )
    
    # 2. 设置 LoRA (针对视觉和语言层)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # 3. 数据集
    train_dataset = SurgicalVideoDataset("/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/train_dataset.jsonl", processor, "./")
    
    # 4. 训练参数
    args = TrainingArguments(
        output_dir="./output/qwen3_vl_surg",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,            # 适当调高起始学习率 (尝试 5e-5)
        lr_scheduler_type="cosine",    # 使用余弦退火，能有效跳出平台期
        warmup_ratio=0.1,              # 增加预热比例，稳定初期梯度
        num_train_epochs=3,
        bf16=True,
        save_strategy="steps",
        save_steps=200,
        optim="paged_adamw_8bit",   # 分页优化器
        weight_decay=0.01,          # 增加权重衰减，防止过拟合
        logging_steps=10
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()

    # 7. 训练结束，保存最后的 LoRA 权重
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_weights"))
    print(f"训练完成！LoRA 权重已保存至 {OUTPUT_DIR}/final_lora_weights")

if __name__ == "__main__":
    train()