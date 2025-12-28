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


if __name__ == "__main__":
    # Run examples
    example_advanced_usage()

