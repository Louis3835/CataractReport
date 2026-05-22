import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_and_processor(model_path):
    print(f"正在从 {model_path} 加载基础模型...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 【核心修复】：解决多卡分布式训练冲突
    # 获取当前进程绑定的显卡 ID。如果不使用分布式，默认回退到 cuda:0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} 
    
    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map, 
        trust_remote_code=True,  # 关键：有了它，AutoModel 会自己去找正确的类
        attn_implementation="sdpa" 
    )
    
    # 开启梯度检查点，用计算时间换显存空间，处理视频 Token 必备
    model.gradient_checkpointing_enable()
    
    return model, processor

def setup_lora(model):
    print("正在配置 LoRA 适配器...")
    
    # 扩大 Target Modules，覆盖更多的线性层，有助于提升多模态推理能力
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=16,                
        lora_alpha=32,      
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 通常要求模型为 fp32 计算某些层，但在 bf16 下 prepare_model_for_kbit_training 也有助于稳定性
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model