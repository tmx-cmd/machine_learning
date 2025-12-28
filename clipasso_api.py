"""
CLIPasso API Wrapper

This module provides an API-like interface for CLIPasso object sketching functionality.
It wraps the CLIPasso-main/run_object_sketching.py script into a callable function.

Usage:
    from clipasso_api import generate_sketch

    result = generate_sketch(
        target_file="path/to/image.jpg",
        num_strokes=16,
        num_iter=2001,
        output_dir="path/to/output"
    )
"""

import os
import sys
import subprocess as sp
import multiprocessing as mp
import time
from shutil import copyfile
import numpy as np
import torch
import argparse
import warnings

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
    """
    Run sketch generation for a single seed.

    Args:
        seed: Random seed for this generation
        wandb_name: Name for this run
        target_path: Path to target image
        output_dir_path: Output directory path
        shared_losses: Shared dictionary for storing losses
        args_config: CLIPassoConfig object with parameters
        use_gpu_flag: Whether to use GPU
        clipasso_path: Path to CLIPasso-main directory
    """
    # Change to CLIPasso directory
    original_cwd = os.getcwd()
    os.chdir(clipasso_path)

    try:
        # Build command
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
        exit_code = sp.run(cmd)

        if exit_code.returncode != 0:
            print(f"Error in subprocess for seed {seed}")
            return

        # Read generated config to calculate loss
        try:
            config_path = f"{output_dir_path}/{wandb_name}/config.npy"
            if os.path.exists(config_path):
                config = np.load(config_path, allow_pickle=True)[()]
                loss_eval = np.array(config['loss_eval'])
                inds = np.argsort(loss_eval)
                # Store result in shared dictionary
                shared_losses[wandb_name] = loss_eval[inds][0]
            else:
                print(f"Config file not found: {config_path}")
        except Exception as e:
            print(f"Error reading config for {wandb_name}: {e}")

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def download_model_if_needed(clipasso_path):
    """
    Download the U2Net model if it doesn't exist.

    Args:
        clipasso_path: Path to CLIPasso-main directory
    """
    model_path = f"{clipasso_path}/U2Net_/saved_models/u2net.pth"

    if not os.path.isfile(model_path):
        print(f"Downloading model to {model_path}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Try to download the model
        try:
            sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                    "-O", "U2Net_/saved_models/u2net.pth"], shell=True, cwd=clipasso_path)
        except Exception as e:
            print(f"Download failed: {e}. Please manually download u2net.pth to {model_path}")
            return False
    return True

def generate_sketch(target_file, num_strokes=16, num_iter=2001, fix_scale=0,
                   mask_object=0, num_sketches=3, use_gpu=None, output_dir=None,
                   clipasso_path=None, multiprocess=True):
    """
    Generate sketches from target image using CLIPasso.

    Args:
        target_file (str): Path to target image file
        num_strokes (int): Number of strokes used to generate the sketch (default: 16)
        num_iter (int): Number of iterations (default: 2001)
        fix_scale (int): Fix scale if target image is not squared (default: 0)
        mask_object (int): Mask background if target image contains background (default: 0)
        num_sketches (int): Number of sketches to generate and choose from (default: 3)
        use_gpu (bool): Whether to use GPU. If None, auto-detect (default: None)
        output_dir (str): Output directory. If None, auto-generate (default: None)
        clipasso_path (str): Path to CLIPasso-main directory. If None, auto-detect (default: None)
        multiprocess (bool): Whether to use multiprocessing (default: True)

    Returns:
        dict: Result containing:
            - success (bool): Whether generation was successful
            - output_dir (str): Output directory path
            - best_sketch_path (str): Path to best sketch file
            - all_sketches (list): List of all generated sketch paths
            - losses (dict): Dictionary of losses for each sketch
    """

    # Auto-detect CLIPasso path
    if clipasso_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clipasso_path = os.path.join(current_dir, "CLIPasso-main", "CLIPasso-main")
        if not os.path.exists(clipasso_path):
            # Try alternative path
            clipasso_path = os.path.join(current_dir, "CLIPasso-main")
            if not os.path.exists(clipasso_path):
                return {
                    "success": False,
                    "error": f"CLIPasso path not found. Expected at {clipasso_path}"
                }

    # Handle relative paths - if target_file is relative and doesn't exist in current dir,
    # try to find it relative to clipasso_path
    if not os.path.isabs(target_file) and not os.path.isfile(target_file):
        # Try relative to clipasso_path
        alt_target_file = os.path.join(clipasso_path, target_file)
        if os.path.isfile(alt_target_file):
            target_file = alt_target_file
        else:
            # Try relative to clipasso_path/target_images
            alt_target_file2 = os.path.join(clipasso_path, "target_images", os.path.basename(target_file))
            if os.path.isfile(alt_target_file2):
                target_file = alt_target_file2
            else:
                return {
                    "success": False,
                    "error": f"Target file does not exist: {target_file} (tried absolute path, relative to CLIPasso dir, and target_images dir)"
                }

    # Final check if target file exists
    if not os.path.isfile(target_file):
        return {
            "success": False,
            "error": f"Target file does not exist: {target_file}"
        }

    # Download model if needed
    if not download_model_if_needed(clipasso_path):
        return {
            "success": False,
            "error": "Failed to download required model"
        }

    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    if not torch.cuda.is_available() and use_gpu:
        use_gpu = False
        print("CUDA is not configured with GPU, running with CPU instead.")

    # Generate output directory
    if output_dir is None:
        target_name = os.path.splitext(os.path.basename(target_file))[0]
        output_dir = f"{clipasso_path}/output_sketches/{target_name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create config object
    config = CLIPassoConfig(
        num_strokes=num_strokes,
        num_iter=num_iter,
        fix_scale=fix_scale,
        mask_object=mask_object,
        num_sketches=num_sketches,
        multiprocess=int(multiprocess and num_sketches > 1)
    )

    # Initialize multiprocessing manager
    manager = mp.Manager()
    losses_all = manager.dict()

    # Generate seeds
    seeds = list(range(0, num_sketches * 1000, 1000))

    # Run sketch generation
    if multiprocess and num_sketches > 1:
        ncpus = min(10, mp.cpu_count())
        print(f"Starting multiprocessing pool with {ncpus} CPUs...")
        P = mp.Pool(ncpus)

        # Launch tasks
        for seed in seeds:
            wandb_name = f"{os.path.splitext(os.path.basename(target_file))[0]}_{num_strokes}strokes_seed{seed}"
            print(f"Launching task: {wandb_name}")

            P.apply_async(run_sketch_generation, args=(
                seed, wandb_name, target_file, output_dir, losses_all, config, use_gpu, clipasso_path
            ))

        P.close()
        P.join()  # Wait for all subprocesses to finish
    else:
        # Single process execution
        for seed in seeds:
            wandb_name = f"{os.path.splitext(os.path.basename(target_file))[0]}_{num_strokes}strokes_seed{seed}"
            print(f"Processing: {wandb_name}")

            run_sketch_generation(
                seed, wandb_name, target_file, output_dir, losses_all, config, use_gpu, clipasso_path
            )

    # Select best sketch
    result = {
        "success": True,
        "output_dir": output_dir,
        "best_sketch_path": None,
        "all_sketches": [],
        "losses": dict(losses_all)
    }

    if len(losses_all) > 0:
        sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
        best_name = list(sorted_final.keys())[0]

        src_file = f"{output_dir}/{best_name}/best_iter.svg"
        dst_file = f"{output_dir}/{best_name}_best.svg"

        if os.path.exists(src_file):
            copyfile(src_file, dst_file)
            result["best_sketch_path"] = dst_file
            print(f"Best sketch saved to: {dst_file}")
        else:
            print("Best sketch file not found (generation may have failed).")

        # Collect all sketch paths
        for name in losses_all.keys():
            sketch_path = f"{output_dir}/{name}/best_iter.svg"
            if os.path.exists(sketch_path):
                result["all_sketches"].append(sketch_path)
    else:
        print("No results found. Check for errors above.")
        result["success"] = False
        result["error"] = "No sketches were generated successfully"

    return result

# Example usage
if __name__ == "__main__":
    # Example: Generate sketch for an image
    result = generate_sketch(
        target_file="path/to/your/image.jpg",
        num_strokes=16,
        num_iter=1000,  # Shorter for testing
        num_sketches=2
    )

    if result["success"]:
        print(f"Sketch generated successfully! Best sketch: {result['best_sketch_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
