import json
import os
import time
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ⚠️ 填入你的 NVIDIA API Key
NVIDIA_API_KEY = "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"

# 创建一个文件锁，确保多线程写入时数据不会错乱
file_lock = threading.Lock()

def process_single_clip(clip_info, processed_videos, client):
    """
    工作线程：处理单个 clip 的逻辑，包含重试机制
    """
    video_rel_path = clip_info["video_path"]
    
    # 如果已经处理过，直接返回 None 跳过
    if video_rel_path in processed_videos:
        return None
        
    with open(clip_info["metadata_path"], 'r', encoding='utf-8') as mf:
        clip_data = json.load(mf)
    
    triplets = clip_data.get("triplets", [])
    triplet_lines = []
    for t in triplets:
        tissue, tool, phase = t["triplet"]
        if tool == "None" and tissue == "None" and phase == "IDLE": 
            continue
        triplet_lines.append(f"Time: {t['timestamp_rel']}s | Phase: {phase} | Tool: {tool} | Tissue: {tissue}")
        
    if not triplet_lines:
        final_report = "No significant surgical operation is detected in this sequence."
    else:
        triplet_str = "\n".join(triplet_lines)
        prompt = (
            "You are an expert ophthalmologist. Here is a sequence of continuous surgical actions from a 5-second cataract surgery clip:\n"
            f"{triplet_str}\n\n"
            "Provide a coherent, objective, and professional surgical description in English (under 50 words). "
            "Output ONLY the description text, no extra formatting or thinking process."
        )
        
        # 🌟 核心改进：引入指数退避的重试机制
        max_retries = 5
        final_report = None
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                  model="meta/llama-3.1-70b-instruct",
                  messages=[{"role": "user", "content": prompt}],
                  temperature=0.3, 
                  max_tokens=128,
                  timeout=30 # 设置超时时间，防止死等
                )
                final_report = completion.choices[0].message.content.strip()
                break # 成功则跳出重试循环
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 5 * (2 ** attempt) # 等待 5s, 10s, 20s...
                    time.sleep(sleep_time)
                else:
                    # 只有在 3 次都失败后才真正放弃
                    print(f"\n❌ [彻底失败] {video_rel_path}: {e}")
                    return None 

    # 组装结果
    entry = {
        "video": video_rel_path,
        "conversations": [
            {"from": "human", "value": "<video>\nPlease observe this short clip of cataract surgery and generate a detailed surgical report."},
            {"from": "gpt", "value": final_report}
        ]
    }
    return entry


def generate_route_b():
    # 建立一个全局 Client，所有线程复用
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    
    catalog_path = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/triplets_catalog.json"
    output_jsonl = "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/train_route_nvidia.jsonl"


    if not os.path.exists(catalog_path):
        print(f"❌ 找不到 Catalog 文件: {catalog_path}")
        return

    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    # =============== 断点续传读取 ===============
    processed_videos = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_videos.add(entry["video"])
                except:
                    pass
        print(f"♻️ 发现已有 {len(processed_videos)} 条记录，将跳过已生成的数据...")

    total_clips = catalog["clips"]
    print(f"开始并发调用 NVIDIA API... (总任务量: {len(total_clips)})")
    
    # 🌟 核心改进：开启多线程池
    # 注意：max_workers 决定了并发数。NVIDIA API 有速率限制 (RPM)。
    # 如果你经常收到 HTTP 429 报错，把这里调小 (如 5)。如果非常顺畅，可以尝试 10 或 20。
    MAX_THREADS = 1
    
    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            # 1. 把所有任务提交进线程池
            future_to_clip = {
                executor.submit(process_single_clip, clip, processed_videos, client): clip 
                for clip in total_clips
            }
            
            # 2. 带有进度条地收集结果
            for future in tqdm(as_completed(future_to_clip), total=len(future_to_clip), desc="Generating"):
                result = future.result()
                if result is not None:
                    # 3. 线程安全地写入硬盘
                    with file_lock:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()

    print(f"\n✅ 所有数据扩写完成！最终文件: {output_jsonl}")

if __name__ == "__main__":
    generate_route_b()