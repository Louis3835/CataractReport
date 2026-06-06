import json
import os
import random
import argparse

# ================= 配置区 =================
CATALOG_PATH = "triplets/triplets_catalog.json"      # 包含元数据路径的目录
TEST_RATIO = 0.1  # 抽取 10% 的病例作为测试集
RANDOM_SEED = 42  # 固定随机种子，保证划分一致，确保 Single 和 Contextual 抽样到完全相同的测试病例进行公平对比
# ==========================================

def split_dataset():
    # 🌟 1. 引入命令行参数解析
    parser = argparse.ArgumentParser(description="病例级数据集物理拆分与源头防卫中心")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "contextual"], 
        required=True,
        help="选择需要拆分的数据模式: 'single' 或 'contextual'"
    )
    args = parser.parse_args()

    input_file = f"train_route_{args.mode}.jsonl"
    output_dir = f"./data_split_{args.mode}"
    
    TRAIN_OUTPUT = os.path.join(output_dir, "train_dataset.jsonl")                 
    TEST_OUTPUT = os.path.join(output_dir, "test_groundtruth.json")                

    if not os.path.exists(input_file):
        print(f"❌ 找不到对应的数据源文件: {input_file}，请确保该文件已在根目录下生成。")
        return

    # 🌟 2. 自动创建相互隔离的独立文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 已激活隔离存储文件夹: {output_dir}")
    print(f"🔍 正在物理扫描 {input_file} 并对非整除边界坏切片进行源头过滤...")
    
    case_dict = {} 
    corrupted_count = 0  # 记录并过滤掉的幻觉切片数量
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            video_path = entry["video"] 
            
            # 🌟核心修改：在数据拆分源头进行硬盘物理存在性与文件大小防卫检查！
            # 如果边界多生成的切片在 videos/ 下不存在，或者大小为0，直接丢弃，不加入训练和测试集
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                corrupted_count += 1
                continue
                
            case_name = os.path.basename(video_path).split('_clip_')[0]
            if case_name not in case_dict:
                case_dict[case_name] = []
            case_dict[case_name].append(entry)
            
    all_cases = list(case_dict.keys())
    total_cases = len(all_cases)
    print(f" 📦 物理检查完毕，共剔除了 {corrupted_count} 个边界缺失/损坏切片。")
    print(f" 📦 实际健康合规的独立手术病例总数: {total_cases} 个。")
    
    # 3. 按 Case 级别进行随机打乱和划分
    random.seed(RANDOM_SEED)
    random.shuffle(all_cases)
    
    test_size = max(1, int(total_cases * TEST_RATIO))
    test_cases = set(all_cases[:test_size])
    train_cases = set(all_cases[test_size:])
    
    print(f" ✂️ 按 {TEST_RATIO*100}% 比例拆分: {len(train_cases)} 个用于训练，{len(test_cases)} 个用于测试。")
    print(f" 📋 测试集病例名单: {list(test_cases)}")

    # 4. 准备获取测试集的三元组 (从 catalog 和 metadata 中读取)
    print("\n 🔗 正在挂载 Triplet 元数据...")
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
        
    video_to_triplets_map = {}
    for clip in catalog["clips"]:
        meta_path = clip["metadata_path"]
        video_rel_path = clip["video_path"]
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta_data = json.load(mf)
                triplets = [t["triplet"] for t in meta_data.get("triplets", [])]
                video_to_triplets_map[video_rel_path] = triplets

    # 5. 执行拆分写入到专属隔离目录
    train_count = 0
    test_records = []
    
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as train_f:
        for case_name, entries in case_dict.items():
            if case_name in train_cases:
                for entry in entries:
                    train_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    train_count += 1
            else:
                for entry in entries:
                    v_path = entry["video"]
                    gpt_report = entry["conversations"][1]["value"]
                    
                    test_record = {
                        "video": v_path,
                        "ground_truth": gpt_report,
                        "prediction": "", 
                        "triplets": video_to_triplets_map.get(v_path, [])
                    }
                    test_records.append(test_record)

    with open(TEST_OUTPUT, 'w', encoding='utf-8') as test_f:
        json.dump(test_records, test_f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 数据集病例级拆分完成！[{args.mode.upper()} 模式]")
    print(f" 📂 干净训练集输出路径: {TRAIN_OUTPUT} ({train_count} 个合规 Clips)")
    print(f" 📂 干净测试集输出路径: {TEST_OUTPUT} ({len(test_records)} 个合规 Clips)")

if __name__ == "__main__":
    split_dataset()