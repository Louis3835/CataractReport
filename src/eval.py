import torch
from transformers import AutoProcessor, Qwen2_5_V_ForCausalLM
from peft import PeftModel
from decord import VideoReader, cpu
import numpy as np

# ================= 配置验证路径 =================
BASE_MODEL_PATH = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/model/Qwen2.5-VL-7B-Instruct" # 你的基座模型路径（请根据实际名字修改，如3B/4B/7B）
LORA_PATH = "./output/qwen3_surg_lora/final_lora_weights"   # 刚训练好的最终权重
TEST_VIDEO_PATH = "./videos/clip_0001.mp4"                   # 挑一个视频片段用来测试

def load_video(video_path, num_frames=4):
    """时序均匀采样视频帧"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [frames] # 返回 list

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("正在加载基座模型和处理器...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 1. 以 bfloat16 加载原始基座模型
    base_model = Qwen2_5_V_ForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", # 验证阶段用单张卡即可
        trust_remote_code=True
    )
    
    # 2. 核心：将你的 LoRA 权重“外挂”融合到基座模型上
    print(f"正在融合 LoRA 权重: {LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval() # 切换到推理模式

    # 3. 准备测试输入数据
    # 提示词必须和你在 train.jsonl 里的 user 提示词完全一致！
    prompt = "Analyze this cataract surgery video clip and identify the structural triplets in the format of (Tool, Action, Tissue)."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": TEST_VIDEO_PATH},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 4. 数据预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_data = load_video(TEST_VIDEO_PATH, num_frames=4) # 保持和训练一致的 4 帧
    
    inputs = processor(
        text=[text],
        videos=video_data,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 5. 让模型生成答案
    print("\n🚀 正在生成手术三元组报告...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False, # 使用 Greedy Search 确保确定性输出
            temperature=0.0
        )
        
    # 6. 解码并打印结果
    # 过滤掉 input_ids 部分，只留下模型新生成的回答
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    print("\n" + "="*40)
    print(f"【测试视频】: {TEST_VIDEO_PATH}")
    print(f"【模型预测结果】:\n{response[0]}")
    print("="*40)

if __name__ == "__main__":
    evaluate()