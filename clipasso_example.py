"""
CLIPasso Multimodal Example (No External Dependencies)
Relies on pure prompt engineering for white background sketch generation.
"""

from clipasso_api import text_to_image
import os
import time

# 导入Stable Diffusion相关库
try:
    from diffusers import StableDiffusionPipeline
    import torch
    STABLE_DIFFUSION_AVAILABLE = True
except ImportError:
    STABLE_DIFFUSION_AVAILABLE = False
    print("Warning: Stable Diffusion not available. Ablation study will be limited.")

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
    import sys

    # 检查命令行参数，决定运行哪个实验
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        # 运行消融实验
        ablation_study()
    else:
        # 运行原始实验
        print("运行原始实验系列...")
        print("如需运行消融实验，请使用: python clipasso_example.py ablation")

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

def ablation_study():
    """
    消融实验：比较三种方法生成图片的效果

    实验组别：
    1. 直接使用Stable Diffusion生成图片
    2. 使用CLIPasso项目生成图片（不使用prompt工程）
    3. 使用CLIPasso项目生成图片（使用prompt工程）
    """

    if not STABLE_DIFFUSION_AVAILABLE:
        print("❌ Stable Diffusion不可用，无法进行消融实验")
        print("请安装: pip install diffusers torch accelerate transformers")
        return

    # 配置实验参数
    base_prompt = "一只可爱的小猫"
    output_base_dir = r"E:\mllab\machine_learning\ablation_study"

    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    print("=" * 80)
    print("🎯 消融实验开始")
    print(f"📝 基础提示词: {base_prompt}")
    print(f"📁 输出目录: {output_base_dir}")
    print("=" * 80)

    # 初始化Stable Diffusion模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    # ========== 实验组1: 直接Stable Diffusion生成 ==========
    print("\n🔬 实验组1: 直接Stable Diffusion生成")
    print("-" * 50)

    group1_dir = os.path.join(output_base_dir, "group1_stable_diffusion_only")
    os.makedirs(group1_dir, exist_ok=True)

    print(f"生成图片: {base_prompt}")

    start_time = time.time()
    sd_image = pipe(
        prompt=base_prompt,
        negative_prompt="blurry, low quality, deformed, ugly",
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512
    ).images[0]

    sd_output_path = os.path.join(group1_dir, "stable_diffusion_direct.png")
    sd_image.save(sd_output_path)
    end_time = time.time()

    print(f"✅ 完成! (耗时: {end_time - start_time:.2f}s)")
    print(f"📄 保存路径: {sd_output_path}")

    # ========== 实验组2: CLIPasso项目（不使用prompt工程）==========
    print("\n🔬 实验组2: CLIPasso项目（不使用prompt工程）")
    print("-" * 50)

    group2_dir = os.path.join(output_base_dir, "group2_clipasso_no_prompt_engineering")

    print(f"生成素描: {base_prompt} (基础提示词，无额外工程)")

    start_time = time.time()
    result2 = text_to_image(
        prompt=base_prompt,  # 只使用基础提示词
        negative_prompt="",  # 不使用负向提示词
        output_dir=group2_dir,
        num_strokes=16,      # 基础笔画数
        num_iter=1000,       # 基础迭代次数
        use_gpu=True,
        multiprocess=False
    )
    end_time = time.time()

    if result2["success"]:
        print(f"✅ 完成! (耗时: {end_time - start_time:.2f}s)")
        print(f"📄 素描路径: {result2['best_sketch_path']}")
        print(f"📄 基础图像路径: {result2.get('base_image_temp_path', 'N/A')}")
    else:
        print(f"❌ 失败: {result2.get('error', '未知错误')}")

    # ========== 实验组3: CLIPasso项目（使用prompt工程）==========
    print("\n🔬 实验组3: CLIPasso项目（使用prompt工程）")
    print("-" * 50)

    group3_dir = os.path.join(output_base_dir, "group3_clipasso_with_prompt_engineering")

    # 使用精心设计的prompt工程
    engineered_prompt = (
        f"{base_prompt}，写实风格，高清细节，"
        "白色背景，干净简洁，专业插画，高质量，"
        "sharp focus, highly detailed, professional illustration"
    )

    engineered_negative = (
        "blurry, low quality, deformed, ugly, extra limbs, "
        "poor anatomy, watermark, text, signature, cartoon, anime"
    )

    print(f"生成素描: {engineered_prompt}")
    print(f"负向提示: {engineered_negative}")

    start_time = time.time()
    result3 = text_to_image(
        prompt=engineered_prompt,
        negative_prompt=engineered_negative,
        output_dir=group3_dir,
        num_strokes=32,      # 更多笔画
        num_iter=2001,       # 更多迭代
        fix_scale=1,         # 固定比例
        mask_object=1,       # 遮罩背景
        use_gpu=True,
        multiprocess=False
    )
    end_time = time.time()

    if result3["success"]:
        print(f"✅ 完成! (耗时: {end_time - start_time:.2f}s)")
        print(f"📄 素描路径: {result3['best_sketch_path']}")
        print(f"📄 基础图像路径: {result3.get('base_image_temp_path', 'N/A')}")
    else:
        print(f"❌ 失败: {result3.get('error', '未知错误')}")

    # ========== 实验总结 ==========
    print("\n" + "=" * 80)
    print("📊 消融实验总结")
    print("=" * 80)

    print("实验组1 (Stable Diffusion直接生成):")
    print(f"  路径: {sd_output_path}")
    print("  特点: 直接生成彩色图像，无素描转换")

    print("\n实验组2 (CLIPasso无prompt工程):")
    if result2["success"]:
        print(f"  素描路径: {result2['best_sketch_path']}")
        print("  特点: 基础提示词，标准参数设置")
    else:
        print("  状态: 生成失败")

    print("\n实验组3 (CLIPasso有prompt工程):")
    if result3["success"]:
        print(f"  素描路径: {result3['best_sketch_path']}")
        print("  特点: 精心设计的提示词，更高质量参数")
    else:
        print("  状态: 生成失败")

    print(f"\n📁 所有结果保存在: {output_base_dir}")
    print("🎯 实验完成！请比较三个实验组的结果。")


# 运行消融实验的示例代码：
#
# # 方法1: 命令行运行
# python clipasso_example.py ablation
#
# # 方法2: 在Python代码中调用
# from clipasso_example import ablation_study
# ablation_study()