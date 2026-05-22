import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 极简加载：使用官方建议的加载方式
model_path = "/media/zack/Data/Data/models/Qwen/Qwen3-4B"
print("⏳ 正在加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
).eval()

# 2. 准备一张你电脑上的随便什么图片 (路径自己改一下)
image_path = "test_image.jpg" # 确保目录下有一张图
if not os.path.exists(image_path):
    # 如果没有图，生成一张纯黑图片测试
    Image.new('RGB', (224, 224), color='red').save("test_image.jpg")
    print("⚠️ 未找到图片，已生成一张红色测试图。")

# 3. 构造最简多模态输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "test_image.jpg"},
            {"type": "text", "text": "What color is this image?"}
        ]
    }
]

# 4. 推理
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

print("🚀 正在推理...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)

output = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n✅ 模型回答: {output[0]}")