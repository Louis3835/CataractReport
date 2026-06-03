import os
import json
import glob

# ================= 1. 配置区 =================
METADATA_DIR = "./triplets/metadata"
ERROR_LOG_OUTPUT = "./invalid_triplets_log.json"

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

def verify_triplets():
    if not os.path.exists(METADATA_DIR):
        print(f"❌ 找不到目录: {METADATA_DIR}")
        return

    json_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    if not json_files:
        print(f"⚠️ 在 {METADATA_DIR} 中没有找到任何 JSON 文件。")
        return

    error_records = []
    total_triplets = 0
    error_count = 0

    print(f"🔍 开始扫描 {len(json_files)} 个 Clip 元数据文件...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                clip_data = json.load(f)
        except Exception as e:
            print(f"❌ 无法读取文件 {file_path}: {e}")
            continue

        triplets_list = clip_data.get("triplets", [])

        for item in triplets_list:
            total_triplets += 1
            triplet = item.get("triplet", [])
            rule_is_valid = item.get("rule_is_valid", True)  # 如果没有这个字段，默认认为是有效的
            
            if len(triplet) != 3:
                continue
                
            tissue, tool, phase = triplet
            
            # 🌟【适配新结构】：获取时间段起点和终点
            t_start = item.get("timestamp_start", 0)
            t_end = item.get("timestamp_end", 0)

            if phase not in VALID_PHASE_TOOL_MAPPING:
                error_count += 1
                error_records.append({
                    "file": os.path.basename(file_path),
                    "timestamp_start": t_start,
                    "timestamp_end": t_end,
                    "error_type": "Unknown Phase",
                    "rule_is_valid": rule_is_valid,
                    "triplet": triplet,
                    "description": f"阶段 '{phase}' 不在有效列表中。"
                })
                continue

            allowed_tools = VALID_PHASE_TOOL_MAPPING[phase]
            # 🌟【逻辑修复】：如果医生规定该阶段没有工具（空列表），那么模型输出 None 是完全合法的
            # if not allowed_tools:
            #     allowed_tools = ["None"]

            if tool not in allowed_tools:
                error_count += 1
                error_records.append({
                    "file": os.path.basename(file_path),
                    "timestamp_start": t_start,
                    "timestamp_end": t_end,
                    "error_type": "Invalid Tool for Phase",
                    "rule_is_valid": rule_is_valid,
                    "triplet": triplet,
                    "description": f"Phase '{phase}' should not have tool '{tool}'. Allowed tools: {allowed_tools}"
                })

    with open(ERROR_LOG_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_triplets_checked": total_triplets,
                "total_errors_found": error_count,
                "error_rate": f"{(error_count / total_triplets * 100):.2f}%" if total_triplets > 0 else "0%"
            },
            "errors": error_records
        }, f, indent=4, ensure_ascii=False)

    print("\n✅ 扫描完成！详细日志已保存。")

if __name__ == "__main__":
    verify_triplets()