"""
CLIPasso Multimodal Example (No External Dependencies)
Relies on pure prompt engineering for white background sketch generation.
"""

from clipasso_api import text_to_image
import os
import time

def run_experiment(prompt, complexity, style, output_subdir):
    """Helper function to run a single experiment"""
    # 请确保这是你想要的输出路径
    base_output_dir = r"E:\mllab\machine_learning\final_project_output_prompts_only"
    output_dir = os.path.join(base_output_dir, output_subdir)
    
    print(f"\n--- Starting Experiment: Style=[{style}], Complexity=[{complexity}] ---")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    result = text_to_image(
        prompt=prompt,
        complexity=complexity,
        style=style,
        output_dir=output_dir,
        multiprocess=False
    )
    end_time = time.time()
    
    if result["success"]:
        print(f"✓ Success! (Time: {end_time - start_time:.2f}s)")
        print(f"  [Base Image]:  {result['base_image_path']}")
        print(f"  [Final Sketch]: {result['best_sketch_path']}")
        print("-" * 60)
    else:
        print(f"✗ Failed: {result.get('error')}")

if __name__ == "__main__":
    # 实验 1: 猫 (高复杂度，写实风格)
    # 我们通过 Prompt 强制要求 "isolated on solid white background"
    run_experiment(
        prompt="A full body shot of a cute fluffy white persian cat sitting, looking at camera",
        complexity="high",
        style="realistic", 
        output_subdir="exp1_cat_pure_prompt"
    )

    # 实验 2: 动漫风格 (中复杂度)
    run_experiment(
        prompt="A magical warrior girl with a sword",
        complexity="medium",
        style="anime",
        output_subdir="exp2_anime_pure_prompt"
    )

    # 实验 3: 简单图标 (低复杂度)
    run_experiment(
        prompt="An icon of a coffee cup",
        complexity="low",
        style="default",
        output_subdir="exp3_icon_pure_prompt"
    )