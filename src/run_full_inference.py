import os
import json
import torch
import argparse
from tqdm import tqdm
from openai import OpenAI
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ==================== 1. 路径硬编码配置 ====================
BASE_MODEL_PATH = "/media/zack/Data/Data/models/Qwen3-VL-4B-Instruct"
PROJECT_ROOT = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport"
TEST_DIR = ""
TEST_JSON_PATH = ""
PREDICTION_OUTPUT = ""
# =========================================================


def configure_test_paths(mode):
    """根据数据拆分模式切换测试集目录。"""
    global TEST_DIR, TEST_JSON_PATH, PREDICTION_OUTPUT

    TEST_DIR = os.path.join(PROJECT_ROOT, f"data_split_{mode}")
    TEST_JSON_PATH = os.path.join(TEST_DIR, "test_groundtruth.json")
    PREDICTION_OUTPUT = os.path.join(TEST_DIR, "test_predictions.json")


def resolve_lora_path(mode):
    """根据训练模式定位对应的 LoRA 权重目录。"""
    return os.path.join(PROJECT_ROOT, "output", "qwen3_4b_vl_surg", mode, "final_lora_weights")

def clean_llm_sse_output(raw_text):
    """专门针对本地服务器流式泄露的文本剥壳器"""
    if "data: {" not in raw_text:
        return raw_text # 如果本来就是干净文本，直接返回
        
    cleaned_text = ""
    for line in raw_text.split('\n'):
        line = line.strip()
        if line.startswith("data: {"):
            try:
                # 剥掉 "data: " 前缀并转为字典
                chunk = json.loads(line[6:])
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        cleaned_text += delta["content"]
            except Exception:
                pass
    return cleaned_text.strip()

def phase_1_clip_inference(model, processor, test_data):
    """阶段一：视频到文本 (Video-to-Text) 的局部推理"""
    print("\n🚀 阶段一：启动本地大模型进行局部切片推理...")
    
    if os.path.exists(PREDICTION_OUTPUT):
        with open(PREDICTION_OUTPUT, 'r', encoding='utf-8') as f:
            completed_data = json.load(f)
            if len(completed_data) == len(test_data):
                print(f"📦 监测到完整的阶段一缓存，共 {len(completed_data)} 条记录，直接跳过生成。")
                return completed_data
            else:
                print(f" 📂 监测到部分完成的缓存 ({len(completed_data)}/{len(test_data)})，将自动实施断点续推...")
                completed_map = {item["video"]: item for item in completed_data}
    else:
        completed_map = {}

    results = []
    
    for item in tqdm(test_data, desc="VLM Inferring Clips"):
        v_path = item["video"]
        
        # 🌟 1. 断点续推：如果这个路径以前成功跑出过结果，直接用缓存
        if v_path in completed_map and completed_map[v_path]["prediction"] != "⚠️ [视频不存在或损坏，已跳过]":
            results.append(completed_map[v_path])
            continue
            
        video_full_path = v_path if os.path.isabs(v_path) else os.path.join(PROJECT_ROOT, v_path)
        
        # 🌟 2. 核心边界防御：如果硬盘上根本没有这个视频，或者文件大小为 0（边界溢出的空切片）
        if not os.path.exists(video_full_path) or os.path.getsize(video_full_path) == 0:
            print(f"\n⚠️ 发现边界溢出或不存在的物理文件: {v_path}，已为您自动跳过并标记。")
            item["prediction"] = "⚠️ [视频不存在或损坏，已跳过]"
            results.append(item)
            
            # 实时增量保存，防止进度丢失
            with open(PREDICTION_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            continue

        # 构造对话模板
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_full_path, "max_pixels": 360 * 480},
                    {"type": "text", "text": "Observe this surgery clip and generate a report."}
                ]
            }
        ]

        # 🌟 3. 兜底异常保护：防止其他偶发损坏视频引起崩溃
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
            item["prediction"] = output_text.strip()

        except Exception as e:
            print(f"\n❌ 警告: 视频 {v_path} 处理时发生未知错误: {str(e)}")
            item["prediction"] = "⚠️ [视频不存在或损坏，已跳过]"

        results.append(item)
        
        # 增量落盘保存
        with open(PREDICTION_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return results


def run_rule_smoothing(predictions_data):
    """后处理方案一：基于 Python 状态机滑动窗口平滑"""
    print("\n⚡ [Pipeline 阶段二] 正在执行【方案一：Python规则时序平滑】...")
    output_file = os.path.join(TEST_DIR, "final_report_rule_smoothed.json")
    
    cases_dict = {}
    for item in predictions_data:
        filename = os.path.basename(item["video"])
        case_id = filename.split('_clip_')[0] if "_clip_" in filename else "Cataract_Case"
        try: clip_idx = int(filename.split('clip_')[1].split('.')[0])
        except ValueError: clip_idx = 0
            
        if case_id not in cases_dict: cases_dict[case_id] = []
        cases_dict[case_id].append({"idx": clip_idx, "pred": item.get("prediction", "").strip()})

    final_reports = {}
    for case_id, clips in cases_dict.items():
        clips = sorted(clips, key=lambda x: x["idx"])
        if not clips: continue
        
        # 滑动窗口去除突变状态噪声
        smoothed_preds = []
        for i in range(len(clips)):
            current_pred = clips[i]["pred"]
            if 0 < i < len(clips) - 1:
                prev_pred = clips[i-1]["pred"]
                next_pred = clips[i+1]["pred"]
                
                def get_phase(text):
                    for kw in ["capsulorhexis", "hydrodissection", "phacoemulsification", "irrigation"]:
                        if kw in text.lower(): return kw
                    return "idle"
                
                if get_phase(prev_pred) == get_phase(next_pred) and get_phase(prev_pred) != "idle":
                    if get_phase(current_pred) == "idle" or len(current_pred) < 80:
                        current_pred = f"The surgeon steadily advanced the {get_phase(prev_pred)} procedure. {next_pred}"
            smoothed_preds.append(current_pred)

        # 宏观跨度合并
        report_chunks = []
        last_phase = None
        for idx, pred in enumerate(smoothed_preds):
            current_phase = "idle"
            for kw in ["capsulorhexis", "hydrodissection", "phacoemulsification", "irrigation"]:
                if kw in pred.lower(): current_phase = kw; break
            
            if current_phase == last_phase and idx > 0:
                cleaned_pred = pred.replace("During the capsulorhexis phase, ", "").replace("During capsulorhexis, ", "")
                cleaned_pred = cleaned_pred.replace("During the hydrodissection phase, ", "")
                report_chunks[-1]["content"] += " " + cleaned_pred
                report_chunks[-1]["end_time"] = (idx + 1) * 5
            else:
                report_chunks.append({"start_time": idx * 5, "end_time": (idx + 1) * 5, "content": pred})
                last_phase = current_phase

        report_text = f"==================================================\n" \
                      f"    CLINICAL CATARACT REPORT (ROUTE: RULE SMOOTHED)\n" \
                      f"    Case Identifier: {case_id}\n" \
                      f"==================================================\n\n"
        for chunk in report_chunks:
            report_text += f"[{chunk['start_time']}s - {chunk['end_time']}s] -----------------------------------\n{chunk['content']}\n\n"
        report_text += "--- End of Rule-Based Clinical Record ---"
        final_reports[case_id] = report_text

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, indent=4, ensure_ascii=False)
    print(f"🎉 报告生成成功！保存在: {output_file}")


def run_llm_distillation(predictions_data, api_key):
    """后处理方案二：调用本地 Qwen3.5-35B 服务器进行医学文本重构"""
    print("\n🧠 [Pipeline 阶段二] 正在执行【方案二：本地35B大模型医学润色】...")
    output_file = os.path.join(TEST_DIR, "final_report_llm_distilled.json")
    
    client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key=api_key)
    
    cases_raw_timelines = {}
    for item in predictions_data:
        filename = os.path.basename(item["video"])
        case_id = filename.split('_clip_')[0] if "_clip_" in filename else "Cataract_Case"
        try: clip_idx = int(filename.split('clip_')[1].split('.')[0])
        except ValueError: clip_idx = 0
            
        if case_id not in cases_raw_timelines: cases_raw_timelines[case_id] = []
        cases_raw_timelines[case_id].append({
            "time": f"[{clip_idx * 5}s - {(clip_idx + 1) * 5}s]", "raw_text": item.get("prediction", "").strip()
        })

    final_reports = {}
    for case_id, timeline_entries in tqdm(cases_raw_timelines.items(), desc="LLM Distilling"):
        sorted_entries = sorted(timeline_entries, key=lambda x: int(x["time"].split('s -')[0].replace('[', '')))
        formatted_timeline_str = "".join([f"{e['time']} Description: {e['raw_text']}\n" for e in sorted_entries])

        system_prompt = (
            "You are an expert clinical ophthalmologist compiling a comprehensive Cataract Surgical Report.\n"
            "I will provide you with a chronological timeline of raw captions generated by a vision model.\n"
            "Smooth out the timing flickers (e.g. repetitive descriptions or fake brief idle phase), group matching continuous phases into broader clear time blocks, and ensure an authoritative clinical tone. Do not fabricate facts."
        )
        user_prompt = f"Patient Case Identifier: {case_id}\n\nRaw Temporal Log:\n{formatted_timeline_str}\n\nPlease generate the final clean report:"

        try:
            completion = client.chat.completions.create(
                model="current",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.2, max_tokens=1024
            )
            
            raw_output = ""
            if isinstance(completion, str):
                raw_output = completion.strip()
            elif hasattr(completion, 'choices'):
                raw_output = completion.choices[0].message.content.strip()
            elif isinstance(completion, dict) and 'choices' in completion:
                raw_output = completion['choices'][0]['message']['content'].strip()
            else:
                raw_output = str(completion).strip()

            # 将乱码外壳剥除，提炼出完美的纯医学报告文本！
            final_reports[case_id] = clean_llm_sse_output(raw_output)
        
        except Exception as e:
            final_reports[case_id] = f"Error processing via LLM: {str(e)}"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, indent=4, ensure_ascii=False)
    print(f"🎉 报告生成成功！保存在: {output_file}")


if __name__ == "__main__":
    api_key = "sk-unsloth-25cb794a070224bdf40a8e9fe0a96e37"

    parser = argparse.ArgumentParser(description="一键多模态白内障手术报告推理与重构总控制台")
    parser.add_argument("--mode", type=str, choices=["single", "contextual"], required=True,
                        help="选择要读取的数据拆分目录: 'single' 或 'contextual'")
    
    parser.add_argument("-m", "--smooth_method", type=str, choices=["rule", "llm"], required=True, 
                        help="选择后处理时序重构的方法: 'rule' (规则平滑) 或 'llm' (大模型蒸馏)")
    
    args = parser.parse_args()

    configure_test_paths(args.mode)
    lora_path = resolve_lora_path(args.mode)

    os.makedirs(TEST_DIR, exist_ok=True)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ 找不到对应模式的 LoRA 权重目录: {lora_path}，请先完成 {args.mode} 模式训练")

    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(f"❌ 找不到测试集输入 {TEST_JSON_PATH}，请确认是否执行了 split_dataset.py")

    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 1. 自动执行或者加载阶段一
    print("⏳ 正在初始化 Qwen3-VL-4B 满血全武装架构...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    
    completed_predictions = phase_1_clip_inference(model, processor, test_data)

    # 🌟 2. 释放 VLM 模型占用的显存，给接下来的 35B 推理腾出空间
    del model
    del base_model
    torch.cuda.empty_cache()

    # 3. 根据参数动态执行阶段二
    if args.smooth_method == "rule":
        run_rule_smoothing(completed_predictions)
    elif args.smooth_method == "llm":
        run_llm_distillation(completed_predictions, api_key)