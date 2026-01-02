"""
CLIPasso API Wrapper (Pure Prompt Version)
Removes rembg dependency. Relies solely on Prompt Engineering for white background.
"""

import os
import sys
import subprocess as sp
import multiprocessing as mp
import time
import tempfile
from shutil import copyfile
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import argparse
import warnings

# 导入文生图相关库
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. text_to_image function will not work.")

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class CLIPassoConfig:
    """Configuration class for CLIPasso parameters"""
    def __init__(self, num_strokes=16, num_iter=2001, fix_scale=0, mask_object=0,
                 num_sketches=3, multiprocess=0, colab=False, cpu=False,
                 display=False, gpunum=0):
        self.num_strokes = num_strokes
        self.num_iter = num_iter
        self.fix_scale = fix_scale
        self.mask_object = mask_object
        self.num_sketches = num_sketches
        self.multiprocess = multiprocess
        self.colab = colab
        self.cpu = cpu
        self.display = display
        self.gpunum = gpunum

def run_sketch_generation(seed, wandb_name, target_path, output_dir_path,
                         shared_losses, args_config, use_gpu_flag, clipasso_path):
    """Run sketch generation for a single seed."""
    original_cwd = os.getcwd()
    os.chdir(clipasso_path)

    try:
        cmd = [
            "python", "painterly_rendering.py", target_path,
            "--num_paths", str(args_config.num_strokes),
            "--output_dir", output_dir_path,
            "--wandb_name", wandb_name,
            "--num_iter", str(args_config.num_iter),
            "--save_interval", str(10),
            "--seed", str(seed),
            "--use_gpu", str(int(use_gpu_flag)),
            "--fix_scale", str(args_config.fix_scale),
            "--mask_object", str(args_config.mask_object),
            "--mask_object_attention", str(args_config.mask_object),
            "--display_logs", str(int(args_config.colab)),
            "--display", str(int(args_config.display))
        ]

        # Run subprocess
        sp.run(cmd, check=True, capture_output=True)

        try:
            config_path = f"{output_dir_path}/{wandb_name}/config.npy"
            if os.path.exists(config_path):
                config = np.load(config_path, allow_pickle=True)[()]
                loss_eval = np.array(config['loss_eval'])
                inds = np.argsort(loss_eval)
                shared_losses[wandb_name] = loss_eval[inds][0]
        except Exception as e:
            print(f"Error reading config for {wandb_name}: {e}")

    except sp.CalledProcessError as e:
        print(f"Error in CLIPasso subprocess for seed {seed}: {e.stderr.decode()}")
    finally:
        os.chdir(original_cwd)

def download_model_if_needed(clipasso_path):
    """Download the U2Net model if it doesn't exist."""
    model_path = f"{clipasso_path}/U2Net_/saved_models/u2net.pth"

    if not os.path.isfile(model_path):
        print(f"Downloading model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            sp.run(["curl", "-L", "-o", "U2Net_/saved_models/u2net.pth", 
                    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.pth"], 
                   cwd=clipasso_path, check=True)
        except Exception as e:
            print(f"Download failed: {e}. Please manually download u2net.pth")
            return False
    return True

def generate_sketch(target_file, num_strokes=16, num_iter=2001, fix_scale=0,
                   mask_object=1, num_sketches=1, use_gpu=None, output_dir=None,
                   clipasso_path=None, multiprocess=False):
    """Generate sketches from target image using CLIPasso."""
    
    # Auto-detect CLIPasso path
    if clipasso_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clipasso_path = os.path.join(current_dir, "CLIPasso-main", "CLIPasso-main")
        if not os.path.exists(clipasso_path):
            clipasso_path = os.path.join(current_dir, "CLIPasso-main")
            if not os.path.exists(clipasso_path):
                return {"success": False, "error": f"CLIPasso path not found."}

    # Handle relative paths
    if not os.path.isabs(target_file):
        target_file = os.path.abspath(target_file)

    if not os.path.isfile(target_file):
        return {"success": False, "error": f"Target file does not exist: {target_file}"}

    if not download_model_if_needed(clipasso_path):
        return {"success": False, "error": "Failed to download required model"}

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    if output_dir is None:
        target_name = os.path.splitext(os.path.basename(target_file))[0]
        output_dir = f"{clipasso_path}/output_sketches/{target_name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = CLIPassoConfig(
        num_strokes=num_strokes, num_iter=num_iter, fix_scale=fix_scale,
        mask_object=mask_object, num_sketches=num_sketches,
        multiprocess=int(multiprocess and num_sketches > 1)
    )

    manager = mp.Manager()
    losses_all = manager.dict()
    seeds = list(range(0, num_sketches * 1000, 1000))

    if multiprocess and num_sketches > 1:
        ncpus = min(4, mp.cpu_count())
        P = mp.Pool(ncpus)
        for seed in seeds:
            wandb_name = f"sketch_{num_strokes}str_{seed}"
            P.apply_async(run_sketch_generation, args=(
                seed, wandb_name, target_file, output_dir, losses_all, config, use_gpu, clipasso_path
            ))
        P.close()
        P.join()
    else:
        for seed in seeds:
            wandb_name = f"sketch_{num_strokes}str_{seed}"
            run_sketch_generation(
                seed, wandb_name, target_file, output_dir, losses_all, config, use_gpu, clipasso_path
            )

    result = {
        "success": True, "output_dir": output_dir,
        "best_sketch_path": None, "all_sketches": [], "losses": dict(losses_all)
    }

    if len(losses_all) > 0:
        sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
        best_name = list(sorted_final.keys())[0]
        src_file = f"{output_dir}/{best_name}/best_iter.svg"
        timestamp = datetime.now().strftime("%H%M%S")
        dst_file = f"{output_dir}/best_sketch_{timestamp}.svg"
        
        if os.path.exists(src_file):
            copyfile(src_file, dst_file)
            result["best_sketch_path"] = dst_file
        
        for name in losses_all.keys():
            sketch_path = f"{output_dir}/{name}/best_iter.svg"
            if os.path.exists(sketch_path):
                result["all_sketches"].append(sketch_path)
    else:
        result["success"] = False
        result["error"] = "No sketches were generated successfully"

    return result

def process_prompt_engineering(prompt, style="default"):
    """
    强化版 Prompt 工程：完全依赖提示词来保证白底和构图
    """
    # 核心增强：强制白底、无阴影、主体居中、完整
    # 添加 "flat lighting" 和 "vector style" 有助于减少阴影
    base_enhancer = ", centered shot, full body, isolated on solid white background, flat lighting, no shadows, clean edges, minimalist, vector style"
    
    styles = {
        "default": base_enhancer,
        "anime": base_enhancer + ", anime key visual, studio ghibli style, thick lineart, cel shaded, manga drawing",
        "realistic": base_enhancer + ", highly detailed, realistic texture, sharp focus, 8k",
        "scribble": base_enhancer + ", rough charcoal sketch style, messy lines, hand drawn on paper"
    }
    
    enhanced_prompt = prompt + styles.get(style, base_enhancer)
    return enhanced_prompt

def text_to_image(prompt, negative_prompt="", output_dir="./generated_images",
                 complexity="medium", style="default",
                 use_gpu=None, clipasso_path=None, multiprocess=False):
    """
    改进后的文生图流程 (无 rembg 依赖版)
    """

    # 1. 参数映射
    complexity_map = {
        "low": {"strokes": 16, "iter": 1500},
        "medium": {"strokes": 32, "iter": 1800},
        "high": {"strokes": 48, "iter": 2200}
    }
    params = complexity_map.get(complexity, complexity_map["medium"])
    num_strokes = params["strokes"]
    num_iter = params["iter"]

    if not DIFFUSERS_AVAILABLE:
        return {"success": False, "error": "diffusers library unavailable"}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 生成基础图像 (Stable Diffusion)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Stage 1] Generating base image | Style: {style} | Complexity: {complexity}")
    
    # Prompt 增强
    enhanced_prompt = process_prompt_engineering(prompt, style)
    
    # 负面 Prompt：这是不用 rembg 的关键。必须明确禁止阴影、复杂背景、裁剪
    enhanced_negative = negative_prompt + ", shadow, drop shadow, cast shadow, dark background, complex background, grey background, gradient background, textured background, cropped, out of frame, cut off, partially visible, watermark, text, blurry, artifacts, deformed, ugly"

    print(f"  > Enhanced Prompt: {enhanced_prompt}")

    # 使用 DPM++ 采样器
    scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    sd_pipe = sd_pipe.to(device)
    if device == "cpu": sd_pipe.enable_attention_slicing()
    
    # 生成图像
    base_image = sd_pipe(
        prompt=enhanced_prompt,
        negative_prompt=enhanced_negative,
        num_inference_steps=25,
        guidance_scale=8.0, # 稍微提高相关性，强迫模型听从 "white background"
        width=512, height=512
    ).images[0]

    # 保存基础图像 (这是唯一的图像源了)
    base_image_path = os.path.join(output_dir, f"base_sd_gen_{timestamp}.png")
    base_image.save(base_image_path)
    print(f"  > Base image saved: {base_image_path}")
    
    # 保存到临时文件供 CLIPasso 使用
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_image_path = temp_file.name
        base_image.save(temp_image_path)

    try:
        # 3. 生成素描 (CLIPasso)
        print(f"[Stage 2] Running CLIPasso | Strokes: {num_strokes} | Iterations: {num_iter}")
        
        sketch_result = generate_sketch(
            target_file=temp_image_path,
            num_strokes=num_strokes,
            num_iter=num_iter,
            fix_scale=1,
            mask_object=1, # 仍然开启 Mask，CLIPasso 内部会利用显著性检测
            num_sketches=1,
            use_gpu=use_gpu,
            output_dir=output_dir,
            clipasso_path=clipasso_path,
            multiprocess=multiprocess
        )
        
        if sketch_result["success"]:
            print("✓ Multimodal pipeline completed successfully!")
            return {
                "success": True,
                "prompt": prompt,
                "base_image_path": base_image_path,
                "best_sketch_path": sketch_result["best_sketch_path"]
            }
        else:
            return {"success": False, "error": sketch_result.get('error')}
            
    except Exception as e:
        return {"success": False, "error": f"Pipeline error: {str(e)}"}
    
    finally:
        try: os.unlink(temp_image_path)
        except: pass