import cv2
import json
import os
import glob
import subprocess
from collections import Counter
from ultralytics import YOLO

# ================= 批量流水线配置区 =================
CONFIG = {
    "model_cls_path": "/home/zack/Desktop/workspace/zhaoubuntu/Segmentation/Yolo/runs/classify/train3/weights/best.pt", 
    "model_seg_path": "/home/zack/Desktop/workspace/zhaoubuntu/Segmentation/Yolo/runs/segment/train9/weights/best.pt",
    "source_dir": "/media/zack/Data/Data/data/Cataract/Phase_recognition_dataset/videos",
    
    "output_json": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/triplets_catalog.json",
    "metadata_dir": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/triplets/metadata",
    "videos_dir": "/home/zack/Desktop/workspace/zhaoubuntu/CataractReport/videos",
    
    "target_fps": 2,
    "conf_threshold": 0.5,
    "clip_seconds": 5,
}

# 工具与组织（严格遵从 YOLO Segment 模型原生输出）
TOOLS = ['Phacoemulsification Tip', 'Slit Knife', 'Gauge', 'Lens Injector', 'Incision Knife', 'Katena Forceps', 'Capsulorhexis Forceps', 'Capsulorhexis Cystotome', 'Spatula', 'Irrigation-Aspiration']
TISSUES = ['Lens', 'Pupil', 'Cornea']

# 阶段与允许工具映射字典（完美桥接了 Classify 的 _ 和 Segment 的 -）
VALID_PHASE_TOOL_MAPPING = {
    "Incision": ["Slit Knife", "Incision Knife"],
    "Viscoelastic": ["Gauge", "Irrigation-Aspiration"],
    "Capsulorhexis": ["Capsulorhexis Cystotome", "Capsulorhexis Forceps"],
    "Hydrodissection": ["Gauge"],
    "Phacoemulsification": ["Phacoemulsification Tip"],
    "Irrigation_Aspiration": ["Irrigation-Aspiration"],
    "Capsule Pulishing": ["Irrigation-Aspiration", "Gauge"],  
    "Lens Implantation": ["Lens Injector"],
    "Lens positioning": ["Spatula", "Katena Forceps"],
    "Anterior_Chamber Flushing": ["Gauge", "Irrigation-Aspiration"],
    "Viscoelastic_Suction": ["Gauge", "Irrigation-Aspiration"],
    "Tonifying_Antibiotics": ["Gauge"],
    "IDLE": ["None"]
}

class BatchTripletExtractor:
    def __init__(self):
        self.model_cls = YOLO(CONFIG["model_cls_path"])
        self.model_seg = YOLO(CONFIG["model_seg_path"])
        self.global_catalog = {"source_dir": CONFIG["source_dir"], "clips": []}

        os.makedirs(CONFIG["metadata_dir"], exist_ok=True)
        os.makedirs(CONFIG["videos_dir"], exist_ok=True)

    def physical_split_video(self, video_path, video_name):
        output_pattern = os.path.join(CONFIG["videos_dir"], f"{video_name}_clip_%04d.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-c", "copy", "-f", "segment",
            "-segment_time", str(CONFIG["clip_seconds"]),
            "-reset_timestamps", "1", output_pattern
        ]
        subprocess.run(command, check=True)

    def process_clip_buffer(self, buffer, clip_start_time):
        """🌟 核心函数：对一个 5 秒 Clip 的多帧数据进行【绑定表决】的聚合推理（含平票非空保护）"""
        if not buffer:
            return None

        pair_votes = []
        tissue_counts = Counter()

        # 1. 【帧级绑定与初步过滤】：在每一帧内部，找出合法的 (Phase, Tool) 组合
        for frame in buffer:
            phase = frame["phase"]
            # 获取该帧 Phase 对应的合法工具列表
            allowed_tools = VALID_PHASE_TOOL_MAPPING.get(phase, [])
            
            # 看看这帧画面里，有没有符合当前 Phase 的合法工具
            valid_tools_in_frame = [t for t in frame["tools"] if t in allowed_tools]
            
            if valid_tools_in_frame:
                # 如果有合法工具，把它们绑在一起作为一票（通常一帧只有一个主工具）
                tool = valid_tools_in_frame[0]
            else:
                # 如果这帧的工具跟 Phase 不匹配，或者根本没抓到工具，记为 None
                tool = "None"
                
            # 投出庄严的一票：(阶段, 工具) 绑定组合
            pair_votes.append((phase, tool))
            
            # 组织 (Tissue) 因为不涉及阶段冲突，依然独立投票
            for t in frame["tissues"]:
                tissue_counts[t] += 1

        # 2. 【Clip 级联合表决与平票裁决机制】
        pair_counter = Counter(pair_votes)
        most_common_pairs = pair_counter.most_common() # 获取所有候选组合及其票数排序
        
        # 默认先取最高票的第一个组合
        best_pair, best_pair_count = most_common_pairs[0]
        
        # 🌟【核心升级逻辑】：如果存在多组平票，且当前最高分是 IDLE 或空工具，尝试把机会让给同票数的合法手术操作组合
        if len(most_common_pairs) > 1:
            for pair, count in most_common_pairs[1:]:
                if count < best_pair_count:
                    break # 票数已经降低，退出检查
                
                # 如果当前最高票是 IDLE 或 None，而平票的竞争对手是【具体手术阶段+具体工具】
                if (best_pair[0] == "IDLE" or best_pair[1] == "None") and (pair[0] != "IDLE" and pair[1] != "None"):
                    best_pair = pair # 逆袭成功：强制让合法的具体临床组合胜出
                    break
        
        majority_phase, final_tool = best_pair

        # 3. 软性校验合规性判定
        rule_is_valid = True
        if majority_phase != "IDLE" and final_tool == "None":
            # 如果这 5 秒赢下来的组合是 (某具体阶段, None)，说明这段时间虽然知道在干嘛，但工具一直没看清/或者一直不合规
            rule_is_valid = False

        # 4. 组织独立表决：选这 5 秒内出现频次最高的组织 (最好出现 >= 3 次以防闪烁)
        final_tissue = "None"
        if tissue_counts:
            valid_tissues = {t: count for t, count in tissue_counts.items() if count >= 3}
            if valid_tissues:
                final_tissue = max(valid_tissues, key=valid_tissues.get)

        # 返回 5 秒片段的【唯一且逻辑严密的三元组】
        return {
            "timestamp_start": round(clip_start_time, 2),
            "timestamp_end": round(clip_start_time + CONFIG["clip_seconds"], 2),
            "triplet": [final_tissue, final_tool, majority_phase],
            "rule_is_valid": rule_is_valid,
            "debug_info": {
                "frames_analyzed": len(buffer),
                "pair_distribution": {f"{p[0]} + {p[1]}": count for p, count in pair_counter.items()}, # 直观展示哪些套装在竞争
                "tissue_frequencies": dict(tissue_counts)
            }
        }

    def process_single_video(self, video_path):
        video_name = os.path.basename(video_path).split('.')[0] 
        print(f"\n正在处理视频: {video_name}")
        
        self.physical_split_video(video_path, video_name)
        
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, round(orig_fps / CONFIG["target_fps"]))
        clip_frame_span = max(1, round(orig_fps * CONFIG["clip_seconds"]))
        
        frame_count = 0
        clip_index = 0
        
        # 帧缓冲区
        frame_data_buffer = []
        clip_records = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_clip_index = frame_count // clip_frame_span
            
            # 当跨越 5 秒边界时，触发聚合处理
            if current_clip_index != clip_index:
                clip_start_time = clip_index * CONFIG["clip_seconds"]
                
                clip_triplet = self.process_clip_buffer(frame_data_buffer, clip_start_time)
                if clip_triplet:
                    clip_records.append({
                        "clip_index": clip_index,
                        "video_name": video_name,
                        "start_time": clip_triplet["timestamp_start"],
                        "end_time": clip_triplet["timestamp_end"],
                        "triplets": [clip_triplet] # 整个片段现在只有一条极致平滑的三元组
                    })
                
                clip_index = current_clip_index
                frame_data_buffer = []

            # 每一帧的基础信息提取（纯净识别，不加任何干扰）
            if frame_count % frame_step == 0:
                res_cls = self.model_cls(frame, verbose=False)[0]
                phase = "IDLE"
                if res_cls.probs and float(res_cls.probs.top1conf) > CONFIG["conf_threshold"]:
                    phase = res_cls.names[int(res_cls.probs.top1)]

                res_seg = self.model_seg(frame, verbose=False)[0]
                names_map = getattr(self.model_seg, "names", {})
                
                tools_in_frame = []
                tissues_in_frame = []

                if res_seg.boxes is not None:
                    try:
                        class_ids = res_seg.boxes.cls.int().tolist()
                        confs = res_seg.boxes.conf.tolist()
                    except Exception:
                        class_ids, confs = [], []

                    for cls_id, conf in zip(class_ids, confs):
                        if float(conf) < CONFIG["conf_threshold"]: continue
                        label = names_map.get(int(cls_id), str(cls_id))

                        if label in TOOLS:
                            tools_in_frame.append(label)
                        elif label in TISSUES:
                            tissues_in_frame.append(label)

                frame_data_buffer.append({
                    "phase": phase,
                    "tools": tools_in_frame,
                    "tissues": tissues_in_frame
                })

            frame_count += 1

        cap.release()
        
        # 处理结尾残留
        if frame_data_buffer:
            clip_start_time = clip_index * CONFIG["clip_seconds"]
            clip_triplet = self.process_clip_buffer(frame_data_buffer, clip_start_time)
            if clip_triplet:
                clip_records.append({
                    "clip_index": clip_index,
                    "video_name": video_name,
                    "start_time": clip_triplet["timestamp_start"],
                    "end_time": clip_triplet["timestamp_end"],
                    "triplets": [clip_triplet]
                })

        # 保存该视频的 metadata
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
        print(f"找到 {len(video_files)} 个视频文件，准备启动【基于时序聚合】的全量流水线...")
        
        for v_path in video_files:
            self.process_single_video(v_path)
            
        os.makedirs(os.path.dirname(CONFIG["output_json"]), exist_ok=True)
        with open(CONFIG["output_json"], 'w', encoding='utf-8') as f:
            json.dump(self.global_catalog, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 自动化提取完成！Catalog 已保存至: {CONFIG['output_json']}")

if __name__ == "__main__":
    extractor = BatchTripletExtractor()
    extractor.run_pipeline()