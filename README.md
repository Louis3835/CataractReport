# Cataract Surgery Report Generation System

本项目旨在通过微调多模态大模型（Qwen3-VL-2B-Instruct），实现对手术视频的理解并自动生成临床手术报告。

## 一、目录结构
- `src/`: 核心代码（训练、数据集处理、模型定义）
- `triplets/`: 手术视频特征三元组提取结果及索引
- `videos/`: 物理视频切片存储目录
- `video_to_triplets.py`: 视频数据预处理与特征标注脚本

## 二、执行流程
### 2.1 数据准备: 提取手术片段的三元组。

```
python3 src/video_to_triplets.py 
```

### 2.2 数据集构建: 整理生成train-test文件。

```
python3 src/split_dataset.py 
```

### 2.3 训练: 微调模型。

```
accelerate launch --multi_gpu --num_processes 2 --gpu_ids="0, 1" src/train.py
```

## 三、技术栈
- 视觉理解: Qwen3-VL-2B-Instruct
- 数据预处理: OpenCV, YOLO (Ultralytics), Decord
- 训练框架: HuggingFace Transformers, PEFT (LoRA)