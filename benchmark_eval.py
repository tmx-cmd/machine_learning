"""
SketchBench-10: Automated Evaluation for Sketch Generation
Features:
1. Objective Metric: Stroke Count Consistency (SVG Parsing)
2. Subjective Metric: Semantic & Aesthetic Scoring (Qwen3-VL)
"""

import os
import json
import base64
import time
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from xml.dom import minidom
import numpy as np

# 导入生成模块
from clipasso_api import text_to_image

# ================= 配置部分 =================
API_KEY = "sk-m9y9MfappohpJOyCvh8ZhA"  # 你的API Key
API_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL_NAME = "qwen3vl"

OUTPUT_DIR = r"E:\mllab\machine_learning\sketch_benchmark_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 复杂度对应的预期笔画数 (参考 clipasso_api.py 中的定义)
COMPLEXITY_MAP = {
    "low": 16,
    "medium": 32,
    "high": 48
}

# ================= 数据集 (SketchBench-10) =================
BENCHMARK_PROMPTS = [
    # --- 简单 (Low, 16 strokes) ---
    {"id": 1, "category": "Icon", "prompt": "An icon of a coffee mug", "complexity": "low"},
    {"id": 2, "category": "Icon", "prompt": "An apple fruit", "complexity": "low"},
    {"id": 3, "category": "Object", "prompt": "A simple desk lamp", "complexity": "low"},
    
    # --- 中等 (Medium, 32 strokes) ---
    {"id": 4, "category": "Animal", "prompt": "A cute cat sitting", "complexity": "medium"},
    {"id": 5, "category": "Animal", "prompt": "A flying bird", "complexity": "medium"},
    {"id": 6, "category": "Food", "prompt": "A slice of pizza with pepperoni", "complexity": "medium"},
    {"id": 7, "category": "Plant", "prompt": "A blooming rose flower", "complexity": "medium"},

    # --- 复杂 (High, 48 strokes) ---
    {"id": 8, "category": "Vehicle", "prompt": "A vintage car side view", "complexity": "high"},
    {"id": 9, "category": "Vehicle", "prompt": "A detailed bicycle", "complexity": "high"},
    {"id": 10, "category": "Animal", "prompt": "A galloping horse with details", "complexity": "high"},
]

# ================= 辅助函数 =================

def count_svg_strokes(svg_path):
    """
    解析SVG文件，精确计算笔画数 (<path> 标签数量)
    """
    try:
        if not os.path.exists(svg_path):
            return 0
        
        # 方法1: XML解析
        try:
            doc = minidom.parse(svg_path)
            paths = doc.getElementsByTagName('path')
            return len(paths)
        except:
            # 方法2: 文本正则解析 (备用)
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.count('<path')
    except Exception as e:
        print(f"Error parsing SVG {svg_path}: {e}")
        return 0

def svg_to_png(svg_path, png_path):
    """
    尝试将SVG转换为PNG供大模型评测。
    如果缺少库，则生成一个空白图或返回False。
    """
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        return True
    except ImportError:
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, png_path, fmt="PNG")
            return True
        except ImportError:
            # print("Warning: cairosvg or svglib not installed. Cannot convert SVG for VL model.")
            return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_with_qwen(image_path, prompt, target_complexity):
    """调用Qwen3-VL进行多维度打分"""
    base64_image = encode_image(image_path)
    
    system_prompt = (
        "You are an expert art critic evaluating a computer-generated sketch. "
        "Assess the image strictly based on the provided metrics."
    )
    
    user_text = f"""
    Task: Evaluate this sketch for the prompt: "{prompt}".
    Target Complexity Level: {target_complexity.upper()}.
    
    Please rate (1-5) on these metrics:
    1. Semantic Alignment: Does it look like the object? (5=Perfect, 1=Unrecognizable)
    2. Sketch Esthetics: Is it a clean, artistic sketch (not a photo)? (5=Beautiful strokes, 1=Messy)
    3. Perceived Complexity: Does the visual detail match the target '{target_complexity}'? 
       (If target is LOW, it should be simple/minimal. If HIGH, it should be detailed. 5=Matches perfectly, 1=Complete mismatch)
    
    Return JSON only:
    {{
        "semantic_score": <int>,
        "esthetics_score": <int>,
        "complexity_match_score": <int>,
        "comment": "<short reasoning>"
    }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": user_text}
            ]}
        ],
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {API_KEY}"
        }, json=payload)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
    except Exception as e:
        print(f"API Error: {e}")
    
    return {"semantic_score": 0, "esthetics_score": 0, "complexity_match_score": 0, "comment": "Error"}

# ================= 主程序 =================

def run_benchmark():
    print(f"🚀 Starting SketchBench-10 (Stroke Consistency & VL Eval)")
    results = []
    
    for item in tqdm(BENCHMARK_PROMPTS):
        # 1. 设置路径
        case_name = f"case_{item['id']}_{item['complexity']}"
        case_dir = os.path.join(OUTPUT_DIR, case_name)
        
        # 2. 生成 (Generation)
        # 检查是否已存在 SVG
        svg_path = os.path.join(case_dir, f"best_sketch.svg") # 假设改名或复制逻辑
        # 注意：clipasso_api.py 返回的是 best_sketch_path，可能是带时间戳的
        
        gen_data = None
        
        # 如果没有缓存，则生成
        if not os.path.exists(case_dir) or not any(f.endswith('.svg') for f in os.listdir(case_dir)):
            print(f"\n🎨 Generating: {item['prompt']} ({item['complexity']})")
            gen_data = text_to_image(
                prompt=item['prompt'],
                complexity=item['complexity'],
                style="realistic", # 保持风格一致以控制变量
                output_dir=case_dir,
                multiprocess=False
            )
            if gen_data and gen_data['success']:
                svg_source = gen_data['best_sketch_path']
                # 为了方便后续处理，找到生成的svg
                svg_path = svg_source
            else:
                print("Generation failed.")
                continue
        else:
            # 找到现有的SVG
            svg_files = [f for f in os.listdir(case_dir) if f.endswith('.svg')]
            if svg_files:
                svg_path = os.path.join(case_dir, svg_files[0])
                # 模拟一个成功返回
                gen_data = {"base_image_path": os.path.join(case_dir, "base.png")} 
            else:
                continue

        # 3. 评测指标 A: 笔画一致性 (Objective Stroke Consistency)
        actual_strokes = count_svg_strokes(svg_path)
        target_strokes = COMPLEXITY_MAP[item['complexity']]
        
        # 计算误差率 (Error Rate)
        stroke_diff = abs(actual_strokes - target_strokes)
        # 一致性分数：100% - 归一化误差。如果误差超过目标值，分数为0
        stroke_consistency_score = max(0, 1 - (stroke_diff / target_strokes)) * 5.0 # 映射到 5分制
        
        print(f"   📏 Strokes: Actual={actual_strokes} | Target={target_strokes} | Score={stroke_consistency_score:.2f}/5")

        # 4. 评测指标 B: 视觉质量 (Subjective VL Eval)
        # 需要图片文件。优先用生成的 SVG 转 PNG，如果没有库，则用 Base Image (仅作参考)
        
        eval_image_path = os.path.join(case_dir, "eval_preview.png")
        is_sketch_image = False
        
        if svg_to_png(svg_path, eval_image_path):
            is_sketch_image = True
        elif gen_data and "base_image_path" in gen_data and os.path.exists(gen_data["base_image_path"]):
            # 降级：如果无法转SVG，使用底图评测语义，但在 prompt 里告诉模型这是底图
            eval_image_path = gen_data["base_image_path"]
            # 注意：用底图评测“素描美感”是不准的，所以这里只是权宜之计
        else:
            eval_image_path = None

        vl_scores = {"semantic_score": 0, "esthetics_score": 0, "complexity_match_score": 0}
        
        if eval_image_path:
            # 只有当是素描图时，评测才有意义；如果是底图，我们只参考语义
            vl_scores = evaluate_with_qwen(eval_image_path, item['prompt'], item['complexity'])
            if not is_sketch_image:
                vl_scores['esthetics_score'] = 0 # 惩罚：无法生成素描预览
                vl_scores['comment'] += " (Evaluated on base image due to missing SVG renderer)"

        # 5. 汇总结果
        row = item.copy()
        row.update({
            "actual_strokes": actual_strokes,
            "target_strokes": target_strokes,
            "stroke_consistency_score": stroke_consistency_score, # 硬指标
            "semantic_score": vl_scores['semantic_score'],        # 软指标
            "esthetics_score": vl_scores['esthetics_score'],      # 软指标
            "perceived_complexity_score": vl_scores['complexity_match_score'], # 软指标
            "judge_comment": vl_scores.get('comment', '')
        })
        results.append(row)
        
        # 避免API限流
        time.sleep(1)

    # ================= 报告生成 =================
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "final_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    
    # 打印控制台报告
    print("\n" + "="*80)
    print("📊 SketchBench-10 Final Report")
    print("="*80)
    
    # 显示详细分数
    print(df[["prompt", "complexity", "actual_strokes", "stroke_consistency_score", "perceived_complexity_score"]])
    
    print("-" * 80)
    print(f"🏆 Overall Metrics (Average / 5.0):")
    print(f"   1. Stroke Consistency (Objective):  {df['stroke_consistency_score'].mean():.2f}")
    print(f"   2. Perceived Complexity (Subjective): {df['perceived_complexity_score'].mean():.2f}")
    print(f"   3. Semantic Alignment (AI Judge):   {df['semantic_score'].mean():.2f}")
    print(f"   4. Sketch Esthetics (AI Judge):     {df['esthetics_score'].mean():.2f}")
    print("="*80)
    print(f"📄 Results saved to: {csv_path}")

    # 简单的可视化：复杂度一致性分析
    try:
        plt.figure(figsize=(10, 6))
        # 归一化并比较
        x = range(len(df))
        plt.bar(x, df['stroke_consistency_score'], width=0.4, label='Stroke Count Precision (Code)', align='center')
        plt.bar([i+0.4 for i in x], df['perceived_complexity_score'], width=0.4, label='Perceived Complexity (AI)', align='center')
        plt.xticks([i+0.2 for i in x], [p[:10]+"..." for p in df['prompt']], rotation=45)
        plt.legend()
        plt.title("Complexity Consistency: Objective (Strokes) vs Subjective (AI Vision)")
        plt.ylabel("Score (0-5)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "complexity_analysis.png"))
        print("📈 Chart generated.")
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    run_benchmark()