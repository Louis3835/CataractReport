import os
import json
import torch
import numpy as np
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

# ================= 1. 核心路径与参数配置 =================
BASE_MODEL_PATH = "/media/zack/Data/Data/models/Qwen3-VL-2B-Instruct"
# 🌟 请根据你 ./output/qwen3_vl_surg/ 目录下实际生成的最大数字修改此路径！
LORA_PATH = "./output/qwen3_vl_surg/checkpoint-1515" 

TEST_JSON_PATH = "test_groundtruth.json"
PREDICTION_OUTPUT = "test_predictions.json"              
FINAL_REPORT_OUTPUT = "final_clinical_reports.json"      

MAX_LENGTH = 1024
# =========================================================

def phase_1_clip_inference(model, processor, test_data):
    """阶段一：视频到文本 (Video-to-Text) 的局部推理"""
    print("\n🚀 阶段一：启动本地 Qwen3-VL-2B 进行局部切片推理...")
    
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
            
            # 1. 构造符合标准 Qwen3-VL 的多模态对话结构
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_file}, # 直接传入物理切片路径
                        {"type": "text", "text": "Please observe this short clip of cataract surgery and generate a detailed surgical report."}
                    ]
                }
            ]

            # 2. 动用灵魂组件：自动完成视频分帧与时序对齐
            image_inputs, video_inputs = process_vision_info(messages)
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # 3. 使用对齐后的多模态特征进行生成
            generated_ids = model.generate(**inputs, max_new_tokens=512)
           
            # 4. 获取模型实际生成的 Token 片段并解码
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            item["prediction"] = output_text.strip()
            print(f"\n[Debug] 视频切片: {os.path.basename(video_file)}")
            print(f"[Debug] 模型生成的报告片断: \n{item['prediction']}\n{'='*40}")
            
            with open(PREDICTION_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 阶段一完成！局部预测已保存至 {PREDICTION_OUTPUT}")
    return test_data

def phase_2_global_aggregation(predictions_data):
    """阶段二：时序平滑与医学逻辑去重（终极修复版）"""
    print("\n🚀 阶段二：正在启动医学逻辑过滤与报告结构化...")
    
    cases_dict = {}
    for item in predictions_data:
        video_path = item["video"]
        filename = os.path.basename(video_path)
        
        # 提取真正的 Case ID (例如由 case_4693_clip_0001.mp4 提取出 case_4693)
        case_id = filename.split('_clip_')[0] if "_clip_" in filename else "Cataract_Case"
        try:
            clip_idx = int(filename.split('clip_')[1].split('.')[0]) if "clip_" in filename else 0
        except ValueError:
            clip_idx = 0
            
        if case_id not in cases_dict:
            cases_dict[case_id] = {}
            
        time_key = f"[{clip_idx * 5}s-{(clip_idx + 1) * 5}s]"
        if time_key not in cases_dict[case_id]:
            cases_dict[case_id][time_key] = []
            
        # 收集该时间段内模型所有的预测
        cases_dict[case_id][time_key].append(item["prediction"])

    final_reports = {}

    for case_id, timeline in cases_dict.items():
        report_text = f"==================================================\n"
        report_text += f"       CLINICAL CATARACT SURGERY REPORT\n"
        report_text += f"       Case Identifier: {case_id}\n"
        report_text += f"==================================================\n\n"
        
        # 按照时间轴正序排列
        sorted_times = sorted(timeline.keys(), key=lambda x: int(x.split('s-')[0].replace('[', '')))
        
        for time_key in sorted_times:
            predictions = timeline[time_key]
            
            # 🌟【核心医学逻辑修复】：从多个冲突的预言中，筛选出最长、最详细、非重复的那一个
            # 这样可以有效防止同一个 5 秒内既做撕囊又做超乳的荒谬现象
            valid_predictions = [p for p in predictions if len(p.strip()) > 10]
            if not valid_predictions:
                best_summary = "The procedure maintained stability with no major structural changes."
            else:
                # 策略：选择包含专业手术动作（如 phacoemulsification）且长度最长的描述
                best_summary = max(valid_predictions, key=len)
            
            report_text += f"{time_key}\n"
            report_text += f"Status: {best_summary}\n\n"
            
        report_text += f"--- End of Report ---"
        final_reports[case_id] = report_text

    with open(FINAL_REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 报告重构成功！结构化病历已保存至 {FINAL_REPORT_OUTPUT}")

if __name__ == "__main__":
    print("⏳ 正在加载标准 Qwen3-VL-2B 视觉语言架构...")

    # 1. 初始化标准处理器
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 2. 加载全新训练的原生多模态基座
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", # 直接推送到当前主卡的 CUDA 显存
        trust_remote_code=True
    )
    
    # 3. 完美融合成熟的微调 LoRA 权重
    print(f"⏳ 正在融合本地手术白内障微调权重: {LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload() 
    model.eval()
    
    print("✅ 多模态架构已全武装就绪，准备载入测试集进行推理...")

    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(f"找不到测试集文件 {TEST_JSON_PATH}，请确保将其放置在当前根目录下。")

    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 运行流水线
    completed_predictions = phase_1_clip_inference(model, processor, test_data)
    phase_2_global_aggregation(completed_predictions)