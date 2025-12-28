"""
CLIPasso API Usage Examples

This file demonstrates how to use the CLIPasso API wrapper.
"""

from clipasso_api import generate_sketch
import os

def example_advanced_usage():
    """Advanced usage with custom parameters"""
    print("\n=== Advanced Usage Example ===")

    #记得修改路径
    target_image = r"E:\mllab\machine_learning\CLIPasso-main\CLIPasso-main\target_images\rose.jpeg"
    output_dir = r"E:\mllab\machine_learning\test_output"

    # Alternative: Use relative path (API will auto-resolve)
    # target_image = "target_images/rose.jpeg"  # API will find this in CLIPasso directory

    if not os.path.exists(target_image):
        print(f"Warning: {target_image} does not exist. Please update the path.")
        return

    result = generate_sketch(
        target_file=target_image,
        num_strokes=32,        # More strokes = more detailed sketch
        num_iter=2001,         # Full iterations
        fix_scale=1,           # Fix scale for non-square images
        mask_object=1,         # Mask background
        num_sketches=1,        # Generate 3 sketches
        use_gpu=True,          # Force GPU usage
        output_dir=output_dir,
        multiprocess=True      # Use multiprocessing
    )

    if result["success"]:
        print("✓ Advanced sketch generated successfully!")
        print(f"  Best sketch: {result['best_sketch_path']}")
        print(f"  Number of sketches generated: {len(result['all_sketches'])}")
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")


def example_text_to_image():
    """文生图示例"""
    print("\n=== 文生图示例 ===")
    
    output_dir = r"E:\mllab\machine_learning\test_text_to_image"
    
    result = text_to_image(
        prompt="一只可爱的小猫在花园里玩耍，阳光明媚",
        negative_prompt="模糊，低质量，变形",
        num_strokes=32,        # 更多笔画 = 更精细
        num_iter=2001,         # 完整迭代次数
        fix_scale=1,           # 固定缩放非正方形图片
        mask_object=1,         # 遮罩背景
        num_sketches=1,        # 生成1个素描
        use_gpu=True,          # 强制使用GPU
        output_dir=output_dir,
        multiprocess=False     # 单进程
    )
    
    if result["success"]:
        print("✓ 文生图成功！")
        print(f"  输出目录: {result['output_dir']}")
        print(f"  最佳素描: {result['best_sketch_path']}")
        print(f"  生成的素描数量: {len(result['all_sketches'])}")
    else:
        print(f"✗ 生成失败: {result.get('error', '未知错误')}")

# 在main函数中调用
if __name__ == "__main__":
    # 运行现有示例
    example_advanced_usage()
    
    # 运行文生图示例
    example_text_to_image()

