import json
import os
import time
import threading
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def clean_sse_response(raw_text):
    """专门用来清洗 Unsloth 本地服务器返回的 SSE 碎片流"""
    if "data: {" not in raw_text:
        return raw_text # 如果是正常的纯文本，直接返回
        
    cleaned_text = ""
    for line in raw_text.split('\n'):
        if line.startswith("data: {"):
            try:
                # 剥掉前缀 "data: " 并解析 JSON
                chunk = json.loads(line[6:])
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        cleaned_text += delta["content"]
            except Exception:
                pass
    return cleaned_text.strip()




file_lock = threading.Lock()

class SurgicalReportPipeline:
    def __init__(self, mode="single"):
        """
        初始化大语言模型报告生成流水线
        :param mode: "single" (一阶段，仅当前 Clip) 或 "contextual" (二阶段，带前后文 3 个 Clip)
        """
        self.mode = mode

    def _load_and_format_triplets(self, metadata_path):
        """读取元数据并将三元组格式化为文本"""
        if not os.path.exists(metadata_path):
            return "Data missing."
            
        with open(metadata_path, 'r', encoding='utf-8') as mf:
            clip_data = json.load(mf)
            
        triplets = clip_data.get("triplets", [])
        triplet_lines = []
        for t in triplets:
            tissue, tool, phase = t["triplet"]
            if tool == "None" and tissue == "None" and phase == "IDLE": 
                continue
            t_start = t.get('timestamp_start', 0.0)
            t_end = t.get('timestamp_end', 5.0)
            triplet_lines.append(f"Time: {t_start}s - {t_end}s | Phase: {phase} | Tool: {tool} | Tissue: {tissue}")
            
        return "\n".join(triplet_lines) if triplet_lines else "No significant surgical operation is detected."

    def build_tasks(self, catalog_clips):
        """
        根据指定的 mode，为所有 clip 预先构建好送给 LLM 的 Prompt 任务列表。
        """
        sorted_clips = sorted(catalog_clips, key=lambda x: (x["video_name"], x["clip_index"]))
        
        # 🌟 1. [新增] 预处理：洗掉所有 rule_is_valid == False 的脏数据
        clean_clips = []
        for clip in sorted_clips:
            try:
                with open(clip["metadata_path"], 'r', encoding='utf-8') as f:
                    clip_data = json.load(f)
                    triplets = clip_data.get("triplets", [])
                    # 检查 5 秒切片中的唯一三元组是否合规
                    if triplets and triplets[0].get("rule_is_valid", True) == True:
                        clean_clips.append(clip)
            except Exception:
                pass
                
        print(f"🧹 数据清洗完成：总切片 {len(sorted_clips)} 个，合规切片 {len(clean_clips)} 个。已剔除 {len(sorted_clips) - len(clean_clips)} 个冲突切片。")

        tasks = []
        # 🌟 2. 遍历清洗后的干净数据列表
        for i, clip in enumerate(clean_clips):
            video_rel_path = clip["video_path"]
            curr_str = self._load_and_format_triplets(clip["metadata_path"])

            if self.mode == "single":
                prompt = (
                    "You are an expert ophthalmologist. Here is a sequence of continuous surgical actions from a 5-second cataract surgery clip:\n"
                    f"{curr_str}\n\n"
                    "Provide a coherent, objective, and professional surgical description in English (under 80 words). "
                    "Output ONLY the description text, no extra formatting or thinking process."
                )
            
            elif self.mode == "contextual":
                prev_str = "None (Start of surgery)"
                next_str = "None (End of surgery)"
                
                # 获取前一个 Clip (在干净列表中找，保证前后文也是合规的)
                if i > 0 and clean_clips[i-1]["video_name"] == clip["video_name"]:
                    prev_str = self._load_and_format_triplets(clean_clips[i-1]["metadata_path"])
                    
                # 获取后一个 Clip
                if i < len(clean_clips) - 1 and clean_clips[i+1]["video_name"] == clip["video_name"]:
                    next_str = self._load_and_format_triplets(clean_clips[i+1]["metadata_path"])

                prompt = (
                    "You are an expert ophthalmologist analyzing a 5-second TARGET clip from a cataract surgery. "
                    "To help you understand the surgical flow, the immediate past and future 5-second contexts are provided.\n\n"
                    f"[Previous 5s Clip (Context)]:\n{prev_str}\n\n"
                    f"[TARGET 5s Clip]:\n{curr_str}\n\n"
                    f"[Next 5s Clip (Context)]:\n{next_str}\n\n"
                    "Based on the context, provide a coherent, objective, and professional surgical description in English (under 80 words) for the **TARGET 5s Clip** ONLY. "
                    "Focus on the exact actions happening in the TARGET timeframe, using the contexts only to resolve ambiguity. "
                    "Output ONLY the description text, no extra formatting or thinking process."
                )

            tasks.append({
                "video": video_rel_path,
                "prompt": prompt
            })
            
        return tasks


def process_llm_task(task, processed_videos, client):
    """工作线程：调用 LLM API"""
    video_rel_path = task["video"]
    
    if video_rel_path in processed_videos:
        return None
        
    prompt = task["prompt"]
    max_retries = 5
    final_report = None
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="current",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, 
                max_tokens=128,
                timeout=120 
            )
            
            # 🌟 无论服务器返回什么妖魔鬼怪，一律转成字符串然后清洗！
            raw_output = str(completion)
            if hasattr(completion, 'choices'):
                raw_output = completion.choices[0].message.content
                
            final_report = clean_sse_response(raw_output)

            break # 拿到完美的纯文本报告，成功跳出重试循环！
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 5 * (2 ** attempt) 
                time.sleep(sleep_time)
            else:
                print(f"\n❌ [彻底失败] {video_rel_path}: {e}")
                return None

    entry = {
        "video": video_rel_path,
        "conversations": [
            {"from": "human", "value": "<video>\nPlease observe this short clip of cataract surgery and generate a detailed surgical report."},
            {"from": "gpt", "value": final_report}
        ]
    }
    return entry


def generate_route_b(pipeline_mode):
    client = OpenAI(
        base_url="http://127.0.0.1:8888/v1", 
        api_key="sk-unsloth-9b5b4593bf0f9bf68c47e0e6027fe11f" 
    )
    
    catalog_path = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/triplets_catalog.json"
    
    # 🌟 在这里一键切换模式："single" 或 "contextual"
    #pipeline_mode = "single" 
    
    output_jsonl = f"/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/train_route_{pipeline_mode}.jsonl"

    if not os.path.exists(catalog_path):
        print(f"❌ 找不到 Catalog 文件: {catalog_path}")
        return

    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    processed_videos = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_videos.add(entry["video"])
                except:
                    pass
        print(f"♻️ 发现已有 {len(processed_videos)} 条记录，将跳过...")

    # 1. 实例化流水线并预先构建所有 Prompt 任务
    pipeline = SurgicalReportPipeline(mode=pipeline_mode)
    all_tasks = pipeline.build_tasks(catalog["clips"])
    
    print(f"🚀 开始并发调用 Qwen3.5-35B API... (当前模式: {pipeline_mode}, 总任务量: {len(all_tasks)})")
    
    MAX_THREADS = 2
    
    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            # 2. 将预构建的任务提交给线程池
            future_to_task = {
                executor.submit(process_llm_task, task, processed_videos, client): task 
                for task in all_tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Generating"):
                result = future.result()
                if result is not None:
                    with file_lock:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()

    print(f"\n✅ 所有数据扩写完成！最终文件: {output_jsonl}")


# 🌟 引入 argparse 解析终端命令
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate surgical reports (Single or Contextual)")
    
    # 定义 --mode 参数，限定只能输入 'single' 或 'contextual'，默认值为 'single'
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "contextual"], 
        default="single", 
        help="选择流水线模式：'single' (仅当前切片) 或 'contextual' (带前后文)"
    )
    
    args = parser.parse_args()
    
    # 把终端抓取到的参数传给主函数
    generate_route_b(pipeline_mode=args.mode)