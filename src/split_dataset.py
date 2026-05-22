import json
import os
import random

# ================= 配置区 =================
INPUT_JSONL = "train_route_nvidia.jsonl"             # 你之前千辛万苦跑出来的总数据
CATALOG_PATH = "triplets/triplets_catalog.json"      # 包含元数据路径的目录
TRAIN_OUTPUT = "train_dataset.jsonl"                 # 拆分后的纯净训练集
TEST_OUTPUT = "test_groundtruth.json"                # 拆分后的测试集(带 Ground Truth)

TEST_RATIO = 0.1  # 抽取 10% 的病例作为测试集
RANDOM_SEED = 42  # 固定随机种子，保证每次拆分结果一致，方便复现实验
# ==========================================

def split_dataset():
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ 找不到总数据文件: {INPUT_JSONL}")
        return

    print("🔍 正在解析病例信息...")
    
    # 1. 扫描现有的 JSONL，提取所有独立病例名称
    case_dict = {} # { "case_4693": [clip1_entry, clip2_entry...] }
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            video_path = entry["video"] # e.g., "videos/case_4693_clip_0000.mp4"
            
            # 提取病例名前缀 (case_xxxx)
            case_name = os.path.basename(video_path).split('_clip_')[0]
            
            if case_name not in case_dict:
                case_dict[case_name] = []
            case_dict[case_name].append(entry)
            
    all_cases = list(case_dict.keys())
    total_cases = len(all_cases)
    print(f" 共发现 {total_cases} 个独立完整手术病例。")
    
    # 2. 按 Case 级别进行随机打乱和划分
    random.seed(RANDOM_SEED)
    random.shuffle(all_cases)
    
    test_size = max(1, int(total_cases * TEST_RATIO))
    test_cases = set(all_cases[:test_size])
    train_cases = set(all_cases[test_size:])
    
    print(f" 按 {TEST_RATIO*100}% 比例拆分: {len(train_cases)} 个用于训练，{len(test_cases)} 个用于测试。")
    print(f" 测试集病例名单: {list(test_cases)}")

    # 3. 准备获取测试集的三元组 (从 catalog 和 metadata 中读取)
    print("\n 正在挂载 Triplet 元数据...")
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
        
    video_to_triplets_map = {}
    for clip in catalog["clips"]:
        meta_path = clip["metadata_path"]
        video_rel_path = clip["video_path"]
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta_data = json.load(mf)
                # 提取原始三元组
                triplets = [t["triplet"] for t in meta_data.get("triplets", [])]
                video_to_triplets_map[video_rel_path] = triplets

    # 4. 执行物理拆分写入
    train_count = 0
    test_records = []
    
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as train_f:
        for case_name, entries in case_dict.items():
            if case_name in train_cases:
                # 写入纯净训练集
                for entry in entries:
                    train_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    train_count += 1
            else:
                # 组装测试集的 Ground Truth 字典
                for entry in entries:
                    v_path = entry["video"]
                    gpt_report = entry["conversations"][1]["value"]
                    
                    test_record = {
                        "video": v_path,
                        "ground_truth": gpt_report,
                        "prediction": "", # 留空，等你微调好的模型来填！
                        "triplets": video_to_triplets_map.get(v_path, [])
                    }
                    test_records.append(test_record)

    with open(TEST_OUTPUT, 'w', encoding='utf-8') as test_f:
        json.dump(test_records, test_f, indent=4, ensure_ascii=False)
        
    print("\n✅ 数据集病例级拆分完成！")
    print(f" 训练集 (Train): {train_count} 个 Clips -> {TRAIN_OUTPUT}")
    print(f" 测试集 (Test):  {len(test_records)} 个 Clips -> {TEST_OUTPUT}")
    print(" 论文撰写提示：请在论文中明确写出使用了 Case-level split 避免数据泄露。")

if __name__ == "__main__":
    split_dataset()