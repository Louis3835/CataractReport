import cv2
import json
import os
import glob
import subprocess
from ultralytics import YOLO

# ================= 批量流水线配置区 =================
CONFIG = {
    "model_cls_path": "/home/zack/Desktop/workspace/zhaoubuntu/Segmentation/Yolo/runs/classify/train3/weights/best.pt",
    "model_seg_path": "/home/zack/Desktop/workspace/zhaoubuntu/Segmentation/Yolo/runs/segment/train9/weights/best.pt",
    "source_dir": "/media/zack/Data/Data/data/Cataract/Phase_recognition_dataset/videos", # 所有视频的根目录
    
    "output_json": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/triplets_catalog.json",
    "metadata_dir": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/metadata",
    "videos_dir": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/videos", # 物理视频切片保存目录
    
    "target_fps": 2,
    "conf_threshold": 0.5,
    "clip_seconds": 5,
}

TOOLS = ['Phacoemulsification Tip', 'Slit Knife', 'Gauge', 'Lens Injector', 'Incision Knife', 'Katena Forceps', 'Capsulorhexis Forceps', 'Capsulorhexis Cystotome', 'Spatula', 'Irrigation-Aspiration']
TISSUES = ['Lens', 'Pupil', 'Cornea']

def _top_detection_by_group(r, group_names, names_map, conf_threshold):
    boxes = getattr(r, "boxes", None)
    if boxes is None: return None
    try:
        class_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
    except Exception: return None

    best_item = None
    for cls_id, conf in zip(class_ids, confs):
        if float(conf) < conf_threshold: continue
        label = names_map.get(int(cls_id), str(cls_id)) if isinstance(names_map, dict) else (names_map[int(cls_id)] if int(cls_id) < len(names_map) else str(cls_id))
        if label not in group_names: continue
        item = {"label": label, "confidence": float(conf)}
        if best_item is None or item["confidence"] > best_item["confidence"]:
            best_item = item
    return best_item

class BatchTripletExtractor:
    def __init__(self):
        self.model_cls = YOLO(CONFIG["model_cls_path"])
        self.model_seg = YOLO(CONFIG["model_seg_path"])
        self.global_catalog = {"source_dir": CONFIG["source_dir"], "clips": []}

        os.makedirs(CONFIG["metadata_dir"], exist_ok=True)
        os.makedirs(CONFIG["videos_dir"], exist_ok=True)

    def physical_split_video(self, video_path, video_name):
        """调用 ffmpeg 将视频物理切片"""
        output_pattern = os.path.join(CONFIG["videos_dir"], f"{video_name}_clip_%04d.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-c", "copy", "-f", "segment",
            "-segment_time", str(CONFIG["clip_seconds"]),
            "-reset_timestamps", "1", output_pattern
        ]
        subprocess.run(command, check=True)

    def process_single_video(self, video_path):
        video_name = os.path.basename(video_path).split('.')[0] # 例如 case_4687
        print(f"\n正在处理视频: {video_name}")
        
        # 1. 物理切片
        self.physical_split_video(video_path, video_name)
        
        # 2. YOLO 提取
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, round(orig_fps / CONFIG["target_fps"]))
        clip_frame_span = max(1, round(orig_fps * CONFIG["clip_seconds"]))
        
        frame_count = 0
        clip_index = 0
        current_clip = {"clip_index": clip_index, "video_name": video_name, "start_time": 0.0, "end_time": round(CONFIG["clip_seconds"], 2), "triplets": []}
        clip_records = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_clip_index = frame_count // clip_frame_span
            if current_clip_index != clip_index:
                if current_clip["triplets"]: clip_records.append(current_clip)
                clip_index = current_clip_index
                clip_start_time = clip_index * CONFIG["clip_seconds"]
                current_clip = {"clip_index": clip_index, "video_name": video_name, "start_time": round(clip_start_time, 2), "end_time": round(clip_start_time + CONFIG["clip_seconds"], 2), "triplets": []}

            if frame_count % frame_step == 0:
                timestamp = frame_count / orig_fps
                res_cls = self.model_cls(frame, verbose=False)[0]
                phase = res_cls.names[int(res_cls.probs.top1)] if res_cls.probs and float(res_cls.probs.top1conf) > CONFIG["conf_threshold"] else "IDLE"

                res_seg = self.model_seg(frame, verbose=False)[0]
                names_map = getattr(self.model_seg, "names", {})
                detected_tool = _top_detection_by_group(res_seg, TOOLS, names_map, CONFIG["conf_threshold"])
                detected_tissue = _top_detection_by_group(res_seg, TISSUES, names_map, CONFIG["conf_threshold"])

                tool = detected_tool["label"] if detected_tool else "None"
                tissue = detected_tissue["label"] if detected_tissue else "None"

                current_clip["triplets"].append({
                    "timestamp_abs": round(timestamp, 2),
                    "timestamp_rel": round(timestamp - current_clip["start_time"], 2),
                    "triplet": [tissue, tool, phase]
                })
            frame_count += 1

        cap.release()
        if current_clip["triplets"]: clip_records.append(current_clip)

        # 3. 保存该视频的 metadata
        for clip in clip_records:
            idx = clip["clip_index"]
            clip_filename = f"{video_name}_clip_{idx:04d}"
            meta_path = os.path.join(CONFIG["metadata_dir"], f"{clip_filename}.json")
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(clip, f, indent=4, ensure_ascii=False)
                
            self.global_catalog["clips"].append({
                "video_name": video_name,
                "clip_index": idx,
                "video_path": f"videos/{clip_filename}.mp4",
                "metadata_path": meta_path
            })

    def run_pipeline(self):
        video_files = glob.glob(os.path.join(CONFIG["source_dir"], "*.mp4"))
        print(f"找到 {len(video_files)} 个视频文件，准备启动全量流水线...")
        
        for v_path in video_files:
            self.process_single_video(v_path)
            
        os.makedirs(os.path.dirname(CONFIG["output_json"]), exist_ok=True)
        with open(CONFIG["output_json"], 'w', encoding='utf-8') as f:
            json.dump(self.global_catalog, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 批量提取完成！共生成 {len(self.global_catalog['clips'])} 个片段。Catalog 已保存至: {CONFIG['output_json']}")

if __name__ == "__main__":
    extractor = BatchTripletExtractor()
    extractor.run_pipeline()