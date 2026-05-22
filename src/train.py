import os
import torch
from transformers import TrainingArguments, Trainer
from model import load_model_and_processor, setup_lora
from dataset import SurgicalVideoDataset, collate_fn

# ================= 核心路径与参数配置 =================
MODEL_PATH = "/media/zack/Data/Data/models/Qwen/Qwen3-4B"
JSONL_PATH = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/train_dataset.jsonl"
VIDEO_FOLDER = "./"                     # 包含 videos/clip_xxxx.mp4 的根目录
OUTPUT_DIR = "./output/qwen3_surg_lora"  # 训练结果保存路径

def train():
    # 1. 加载模型和处理器 (来自你之前的 model.py)
    model, processor = load_model_and_processor(MODEL_PATH)
    
    # 2. 挂载 LoRA 适配器
    model = setup_lora(model)
    
    # 显式激活基座模型的梯度检查点（能省下接近 30% - 40% 的反向传播显存）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.get_input_embeddings().register_forward_hook(lambda module, input, output: output.requires_grad_(True))

    # 3. 构建数据集 (来自你之前的 dataset.py)
    # 假设每段5秒视频我们均匀抽 8 帧
    train_dataset = SurgicalVideoDataset(
        jsonl_path=JSONL_PATH,
        processor=processor,
        video_folder=VIDEO_FOLDER,
        num_frames=4,
        max_length=512
    )

    # 4. 设置训练超参数 (针对你的双 A5000 24GB 显存深度优化)
    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     per_device_train_batch_size=1,   # 视频Token极大，单卡Batch设为1，靠梯度累积放大
    #     gradient_accumulation_steps=4,   # 显存优化：每4步更新一次参数，等效 Batch Size = 1 * 4 * 2(双卡) = 8
    #     learning_rate=2e-5,              # LoRA 微调经典学习率
    #     num_train_epochs=5,              # 迭代轮数
    #     bf16=True,                       # A5000原生支持bf16，速度快且稳定
    #     logging_steps=10,                # 每10步打印一次Loss
    #     save_strategy="steps",           # 评估保存策略
    #     save_steps=50,                   # 每50步保存一个检查点
    #     save_total_limit=2,              # 最多保留2个权重，防止撑爆 A5000 服务器的硬盘
    #     gradient_checkpointing=True,     # 极端重要：通过重算激活值节省 70% 显存
    #     ddp_find_unused_parameters=False,# 提高分布式训练效率
    #     report_to="none",                 # 暂时不上传到 weights&biases
    #     dataloader_num_workers=2,             # 数据加载线程数，视 CPU 核心数调整
    # )
    

    # 4. 设置训练超参数 (全量数据版)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=4,   # 显存优化
        learning_rate=2e-5,              
        num_train_epochs=3,              # 数据变多后，3 个 epoch 看是否足够收敛
        bf16=True,                       
        optim="adamw_8bit",              # 8-bit Adam 优化器，进一步降低显存占用
        logging_steps=10,                
        save_strategy="steps",           
        save_steps=200,                  # 每 200 步保存一次
        save_total_limit=3,              # 保留最近的 3 个检查点
        gradient_checkpointing=True,     
        ddp_find_unused_parameters=False,
        report_to="none",                 
        dataloader_num_workers=4,        # 提升数据加载线程，加快视频处理
    )

    # 5. 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn, # 批处理对齐函数
    )

    # 6. 开启训练
    print("框架搭建完毕，双卡 A5000 启动微调...")
    trainer.train()

    # 7. 训练结束，保存最后的 LoRA 权重
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_weights"))
    print(f"训练完成！LoRA 权重已保存至 {OUTPUT_DIR}/final_lora_weights")

if __name__ == "__main__":
    train()