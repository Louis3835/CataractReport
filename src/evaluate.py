import json
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from collections import defaultdict

# 首次运行需要下载 NLTK 依赖
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class CataractMetricsSystem:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4 

    def evaluate_nlg(self, ground_truths, predictions):
        """计算标准 NLP 指标 (BLEU, ROUGE, METEOR)"""
        refs_bleu = [[gt.lower().split()] for gt in ground_truths]
        preds_bleu = [pred.lower().split() for pred in predictions]

        # BLEU Scores
        bleu1 = corpus_bleu(refs_bleu, preds_bleu, weights=(1, 0, 0, 0), smoothing_function=self.smoothie)
        bleu4 = corpus_bleu(refs_bleu, preds_bleu, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothie)

        # ROUGE & METEOR
        rougeL_scores = []
        meteor_scores = []
        
        for gt, pred in zip(ground_truths, predictions):
            rougeL_scores.append(self.rouge_scorer.score(gt, pred)['rougeL'].fmeasure)
            # METEOR 需要 token 列表
            meteor_scores.append(meteor_score([gt.lower().split()], pred.lower().split()))

        return {
            "BLEU-1": round(bleu1 * 100, 2),
            "BLEU-4": round(bleu4 * 100, 2),
            "ROUGE-L": round(np.mean(rougeL_scores) * 100, 2),
            "METEOR": round(np.mean(meteor_scores) * 100, 2)
        }

    def evaluate_clinical_triplets(self, predictions, gt_triplets_list):
        """
        严谨的临床医学实体评估 (Clinical Entity F1 & AP_IVT)
        基于字符串匹配的近似评估，确保模型准确提及了器械和组织
        """
        metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
        strict_matches = 0

        for pred, gt_triplets in zip(predictions, gt_triplets_list):
            pred_lower = pred.lower()
            
            # 提取这一段视频中真实存在的有效实体 (排除 None 和 IDLE)
            true_tissues = set([t[0].lower() for t in gt_triplets if t[0] != "None"])
            true_tools = set([t[1].lower() for t in gt_triplets if t[1] != "None"])
            true_phases = set([t[2].lower() for t in gt_triplets if t[2] != "idle"])

            # 评估 Tool
            for tool in true_tools:
                if tool in pred_lower: metrics["Tool"]["TP"] += 1
                else: metrics["Tool"]["FN"] += 1
            
            # 评估 Tissue
            for tissue in true_tissues:
                if tissue in pred_lower: metrics["Tissue"]["TP"] += 1
                else: metrics["Tissue"]["FN"] += 1

            # Strict Triplet Match (AP_IVT 思想): 这句话里必须同时包含对应的 Tool 和 Tissue 才算全对
            is_strict_match = True
            if not true_tools and not true_tissues:
                # 如果 GT 本身就是空的，且预测里提到了任何手术器械，算作幻觉/匹配失败
                if any(k.lower() in pred_lower for k in ["knife", "forceps", "phaco", "gauge"]):
                    is_strict_match = False
            else:
                for tool in true_tools:
                    if tool not in pred_lower: is_strict_match = False
                for tissue in true_tissues:
                    if tissue not in pred_lower: is_strict_match = False
                    
            if is_strict_match: strict_matches += 1

        # 计算各类别的 Precision, Recall, F1
        results = {}
        for category, counts in metrics.items():
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"{category}_Precision"] = round(precision * 100, 2)
            results[f"{category}_Recall"] = round(recall * 100, 2)
            results[f"{category}_F1"] = round(f1 * 100, 2)

        results["Strict_Triplet_Accuracy"] = round((strict_matches / len(predictions)) * 100, 2)
        return results

    def run(self, json_data_path):
        with open(json_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truths = [item["ground_truth"] for item in data]
        predictions = [item["prediction"] for item in data]
        gt_triplets = [item["triplets"] for item in data]

        print(f"📊 开始全面评估，测试样本总数: {len(data)}")
        
        nlg_res = self.evaluate_nlg(ground_truths, predictions)
        print("\n" + "="*40 + "\n[1] NLG 语义流畅度指标\n" + "-"*40)
        for k, v in nlg_res.items(): print(f"{k}: \t{v}")

        clinical_res = self.evaluate_clinical_triplets(predictions, gt_triplets)
        print("\n" + "="*40 + "\n[2] Clinical 临床医学三元组指标\n" + "-"*40)
        for k, v in clinical_res.items(): print(f"{k}: \t{v}")
        print("="*40)

# ================= 伪数据测试 =================
if __name__ == "__main__":
    dummy_data = [
        {
            "ground_truth": "The surgeon uses the Phacoemulsification Tip to interact with the Lens.",
            "prediction": "In this phase, the Phacoemulsification Tip is utilized to break the Lens.",
            "triplets": [["Lens", "Phacoemulsification Tip", "Phacoemulsification"]]
        },
        {
            "ground_truth": "The Slit Knife is used to make an incision on the Cornea.",
            "prediction": "The surgeon operates on the Cornea.", # 漏掉了 Tool
            "triplets": [["Cornea", "Slit Knife", "Incision"]]
        }
    ]
    with open("test_predictions.json", "w") as f:
        json.dump(dummy_data, f)
        
    evaluator = CataractMetricsSystem()
    evaluator.run("test_predictions.json")