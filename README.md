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

## 三、文件说明：

---

### 项目目录作用说明

* **`CataractReport/`**: 项目根目录。
* **`src/`**: 核心源代码目录。
* `dataset.py`: 定义数据加载逻辑，负责读取视频、进行视觉特征预处理（使用 `qwen-vl-utils`），并将数据转化为模型训练所需的格式。
* `model.py`: 负责模型的加载、架构初始化以及 LoRA 适配器的配置。
* `train.py`: 项目的训练入口文件，通过 `Trainer` 类进行多模态微调，支持分布式训练。


* **`triplets/`**: 预处理后的中间数据目录。
* `metadata/`: 存放每个 5 秒视频切片的详细元数据（三元组提取结果）。
* `triplets_catalog.json`: 索引文件，记录了所有视频切片的路径及其对应的时间轴。


* **`videos/`**: 存放原始手术视频的物理切片，用于训练时实时读取。
* **`video_to_triplets.py`**: 数据预处理脚本，调用 YOLO 模型对视频进行切片并提取“工具-组织-手术阶段”三元组，为训练生成基础标注。
* **`train_dataset.jsonl`**: 整理后的最终训练集，包含了视频路径及预期的多模态指令格式。

---


## 四、技术栈
- 视觉理解: Qwen3-VL-2B-Instruct
- 数据预处理: OpenCV, YOLO (Ultralytics), Decord
- 训练框架: HuggingFace Transformers, PEFT (LoRA)