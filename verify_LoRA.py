# load_simple.py
from transformers import AutoModelForCausalLM
import torch

# 用最原始的方式加载，不要指定任何 VL 架构，看看能不能加载你的 LoRA
model = AutoModelForCausalLM.from_pretrained(
    "/media/zack/Data/Data/models/Qwen/Qwen3-4B", 
    device_map="auto",
    torch_dtype=torch.bfloat16
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "./output/qwen3_surg_lora/final_lora_weights")
print("✅ 纯语言模型架构加载成功！")