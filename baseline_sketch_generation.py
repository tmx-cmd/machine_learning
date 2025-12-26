import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import os

# -------- 配置参数 --------
PROMPT = "a continuous line drawing of a cat, minimalist, vector art, black ink on white background, simple strokes, no shading, clean lines"
NEGATIVE_PROMPT = "photorealistic, complex, shading, texture, blur, noise, detailed background, color, gray, filling"
NUM_SAMPLES = 5  # 生成多个样本进行对比
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# -------- 加载Stable Diffusion模型 --------
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

# 使用DDIM调度器获得更好的结果
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# -------- 创建输出目录 --------
os.makedirs("outputs/baseline", exist_ok=True)

# -------- 生成基准图像 --------
print(f"Generating {NUM_SAMPLES} baseline sketch images...")

for i in range(NUM_SAMPLES):
    print(f"Generating sample {i+1}/{NUM_SAMPLES}...")

    # 设置种子以获得可重复的结果
    generator = torch.Generator(device=DEVICE).manual_seed(42 + i)

    # 生成图像
    with torch.no_grad():
        image = pipe(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=8.5,
            generator=generator
        ).images[0]

    # 保存图像
    output_path = f"outputs/baseline/cat_baseline_{i+1}.png"
    image.save(output_path)
    print(f"Saved: {output_path}")

    # 同时保存为灰度版本（模拟素描效果）
    gray_image = image.convert("L")
    gray_path = f"outputs/baseline/cat_baseline_gray_{i+1}.png"
    gray_image.save(gray_path)
    print(f"Saved grayscale version: {gray_path}")

print("Baseline generation completed!")
print("Check outputs/baseline/ directory for results.")
print(f"Prompt used: '{PROMPT}'")
print(f"Negative prompt: '{NEGATIVE_PROMPT}'")
