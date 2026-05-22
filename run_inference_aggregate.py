import os
import json
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from tqdm import tqdm

# ================= 1. 核心路径与参数配置 =================
BASE_MODEL_PATH = "/media/zack/Data/Data/models/Qwen/Qwen3-4B"
LORA_PATH = "./output/qwen3_surg_lora/final_lora_weights" 
TEST_JSON_PATH = "test_groundtruth.json"
PREDICTION_OUTPUT = "test_predictions.json"              
FINAL_REPORT_OUTPUT = "final_clinical_reports.json"      

NUM_FRAMES = 4
MAX_LENGTH = 512
# =========================================================

def load_video_frames(video_path, num_frames):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except Exception as e:
        print(f"\n[Warning] 无法读取视频 {video_path}: {e}")
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return [Image.fromarray(dummy_frame) for _ in range(num_frames)]

def phase_1_clip_inference(model, processor, test_data):
    """阶段一：视频到文本 (Video-to-Text) 的局部推理"""
    print("\n🚀 阶段一：启动本地 Qwen3 进行局部切片推理...")
    
    # 检查断点续传
    if os.path.exists(PREDICTION_OUTPUT):
        with open(PREDICTION_OUTPUT, 'r', encoding='utf-8') as f:
            completed_data = json.load(f)
            if len(completed_data) == len(test_data):
                print(f"♻️ 发现已完成的预测文件 ({len(test_data)} 条)，跳过推理阶段。")
                return completed_data
            else:
                test_data = completed_data 

    with torch.no_grad():
        for item in tqdm(test_data, desc="Inferring Clips"):
            if item.get("prediction", "") != "":
                continue
                
            video_file = item["video"]
            pixel_values = load_video_frames(video_file, NUM_FRAMES)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pixel_values},
                        {"type": "text", "text": "Please observe this short clip of cataract surgery and generate a detailed surgical report."}
                    ],
                }
            ]

            texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[texts],
                videos=[pixel_values],
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            item["prediction"] = output_text.strip()
            
            with open(PREDICTION_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 阶段一完成！局部预测已保存至 {PREDICTION_OUTPUT}")
    return test_data

def phase_2_global_aggregation(model, processor, predictions_data):
    """阶段二：文本到文本 (Text-to-Text) 的全局聚合，完全复用本地模型"""
    print("\n🚀 阶段二：使用本地 Qwen3 进行全局时序总结 (纯文本模式)...")
    
    cases_dict = {}
    for item in predictions_data:
        video_path = item["video"]
        case_id = os.path.basename(video_path).split('_clip_')[0]
        clip_idx = int(os.path.basename(video_path).split('_clip_')[1].split('.')[0])
        
        if case_id not in cases_dict:
            cases_dict[case_id] = []
            
        cases_dict[case_id].append({
            "time_start": clip_idx * 5,
            "time_end": (clip_idx + 1) * 5,
            "text": item["prediction"]
        })

    final_reports = {}

    with torch.no_grad():
        for case_id, clips in tqdm(cases_dict.items(), desc="Generating Global Reports"):
            clips.sort(key=lambda x: x["time_start"])
            
            timeline_log = ""
            for c in clips:
                # 简单过滤无动作的废话，减轻 4B 模型的阅读负担
                if "No significant surgical operation" not in c["text"] and "IDLE" not in c["text"]:
                    timeline_log += f"[{c['time_start']}s-{c['time_end']}s]: {c['text']}\n"
            
            if not timeline_log.strip():
                final_reports[case_id] = "No active surgical phases were detected in this sequence."
                continue

            # 纯文本 Prompt，要求模型进行润色
            prompt = (
                "You are an expert ophthalmic surgeon. Rewrite the following chronological log of surgical actions "
                "into a single, cohesive, and continuous operative summary paragraph. "
                "Remove timestamps, avoid repetitions, and ensure professional medical terminology.\n\n"
                f"Surgical Log:\n{timeline_log}\n\n"
                "Final Summary Report:"
            )

            # 🌟 纯文本输入模式：不需要传递 video，直接传字符串
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

            # 增加生成长度，适应完整的长文报告
            generated_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            final_reports[case_id] = output_text.strip()

    with open(FINAL_REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 阶段二完成！完整的本地病例级手术报告已保存至 {FINAL_REPORT_OUTPUT}")

if __name__ == "__main__":
    # 统一在主函数加载模型，跨阶段共享，极大节省显存和加载时间
    print("⏳ 正在将大模型加载至双卡 A5000 显存中...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 读取测试集
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 依次执行流水线
    completed_predictions = phase_1_clip_inference(model, processor, test_data)
    phase_2_global_aggregation(model, processor, completed_predictions)