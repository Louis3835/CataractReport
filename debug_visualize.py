import cv2
import os
import json
import re
from ultralytics import YOLO

# ================= 配置区 =================
LOG_PATH = "./invalid_triplets_log.json"
METADATA_DIR = "./triplets/metadata" # 🌟 新增：元数据文件夹路径，用于顺藤摸瓜找视频
SEG_MODEL_PATH = "/home/zack/Desktop/workspace/zhaoubuntu/Segmentation/Yolo/runs/segment/train9/weights/best.pt"
DEBUG_OUT_DIR = "./debug_visualizations"
# =========================================

os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
model_seg = YOLO(SEG_MODEL_PATH)

with open(LOG_PATH, 'r', encoding='utf-8') as f:
    log_data = json.load(f)

errors = log_data.get("errors", [])
print(f"Loaded {len(errors)} error points. Starting dynamic video matching...")

# 用来缓存已经打开的视频捕获器，避免频繁重复打开同一个视频文件，提升效率
video_capture_cache = {}

def draw_multiline_text(img, text, start_x, start_y, max_width, font_scale, color, thickness):
    # 🌟 核心修复：用正则表达式把文本中所有引发乱码的非英文字符（如中文句号、冒号、中文字) 彻底滤掉
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # 把可能因为过滤连续产生的多个空格缩减为一个
    clean_text = ' '.join(clean_text.split())

    words = clean_text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if text_width > max_width and current_line != "":
            lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line.strip())
        
    current_y = start_y
    for line in lines:
        cv2.putText(img, line, (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        current_y += int(30 * font_scale) + 5
    return current_y


# 挑选前 30 个代表性的错误实例进行可视化
for idx, err in enumerate(errors[:30]):
    # 🌟【适配新结构】：读取起始时间
    target_start = err.get("timestamp_start", 0)
    target_end = err.get("timestamp_end", 5.0)
    error_type = err["error_type"]
    triplet = err["triplet"]
    desc = err["description"]
    error_file = err["file"]
    
    meta_file_path = os.path.join(METADATA_DIR, error_file)
    actual_video_path = None
    
    if os.path.exists(meta_file_path):
        try:
            with open(meta_file_path, 'r', encoding='utf-8') as mf:
                meta_data = json.load(mf)
                actual_video_path = meta_data.get("video_path") or meta_data.get("video")
        except Exception as e:
            pass
            
    if not actual_video_path or not os.path.exists(actual_video_path):
        possible_video_name = error_file.replace(".json", ".mp4")
        possible_path = os.path.join("./videos", possible_video_name)
        if os.path.exists(possible_path):
            actual_video_path = possible_path
            
    if not actual_video_path or not os.path.exists(actual_video_path):
        print(f"❌ 跳过第 {idx} 个错误：找不到匹配的物理视频 {error_file}")
        continue

    if actual_video_path not in video_capture_cache:
        video_capture_cache[actual_video_path] = cv2.VideoCapture(actual_video_path)
        
    cap = video_capture_cache[actual_video_path]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 🌟【核心取帧逻辑重构】：直击 5秒 Clip 的正中心 (2.5秒)
    if "_clip_" in error_file:
        relative_time = 2.5  # 在物理切片内部，2.5秒是最具代表性的中点帧
    else:
        # 如果你读的是未切分的长视频，取起止时间的平均值
        relative_time = target_start + ((target_end - target_start) / 2.0)

    frame_no = int(relative_time * fps)
    
    if frame_no >= total_frames:
        frame_no = total_frames - 1
    if frame_no < 0:
        frame_no = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if not ret:
        continue
        
    results = model_seg(frame, verbose=False)[0]
    annotated_frame = results.plot()  
    
    img_height, img_width = annotated_frame.shape[:2]
    max_text_width = img_width - 60  
    
    v_name = os.path.basename(actual_video_path)
    # 🌟【文案更新】：展示 5 秒的时间段，凸显这是一种宏观裁决
    info_text = f"Source: {v_name} | Clip: {target_start}s-{target_end}s | Phase: {triplet[2]} | Tool: {triplet[1]}"
    cv2.putText(annotated_frame, info_text, (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    
    type_text = f"Rule Error: {error_type}"
    cv2.putText(annotated_frame, type_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    desc_text = f"Description: {desc}"
    draw_multiline_text(
        img=annotated_frame, 
        text=desc_text, 
        start_x=30, 
        start_y=65, 
        max_width=max_text_width, 
        font_scale=0.42, 
        color=(255, 255, 255), 
        thickness=1
    )
    
    out_img_path = os.path.join(DEBUG_OUT_DIR, f"err_{idx:02d}_{error_file.replace('.json', '')}.jpg")
    cv2.imwrite(out_img_path, annotated_frame)
    print(f"📸 坏样本图已保存: {out_img_path}")

for cap in video_capture_cache.values():
    cap.release()
    
print(f"\n🔍 检查完成！请打开目录 {DEBUG_OUT_DIR} 查看图片。每一张图现在都与它原本的 triplets 完美对齐了！")