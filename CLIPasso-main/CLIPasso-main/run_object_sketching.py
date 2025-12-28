import sys
import warnings
import time
import argparse
import multiprocessing as mp
import os
import subprocess as sp
from shutil import copyfile
import numpy as np
import torch

# 尝试导入 IPython，如果失败则给出一个友好的提示（防止脚本直接崩溃）
try:
    from IPython.display import Image as Image_colab
    from IPython.display import display, SVG, clear_output
    from ipywidgets import IntSlider, Output, IntProgress, Button
except ImportError:
    print("Warning: IPython not installed. Display features will not work.")
    # 定义空函数防止报错
    def display(*args, **kwargs): pass
    def SVG(*args, **kwargs): pass
    def clear_output(*args, **kwargs): pass
    Image_colab = None

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# -------------------------------------------------------------------------
# 参数定义 (保持在全局，以便子进程也能访问配置)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=16,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=2001,
                    help="number of iterations")
parser.add_argument("--fix_scale", type=int, default=0,
                    help="if the target image is not squared, it is recommended to fix the scale")
parser.add_argument("--mask_object", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")
parser.add_argument("--num_sketches", type=int, default=3,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument("--multiprocess", type=int, default=0,
                    help="recommended to use multiprocess if your computer has enough memory")
parser.add_argument('-colab', action='store_true')
parser.add_argument('-cpu', action='store_true')
parser.add_argument('-display', action='store_true')
parser.add_argument('--gpunum', type=int, default=0)

# 解析参数
args = parser.parse_args()

# -------------------------------------------------------------------------
# 核心函数定义
# -------------------------------------------------------------------------

def run(seed, wandb_name, target_path, output_dir_path, shared_losses, args_config, use_gpu_flag):
    """
    运行绘画的主函数。
    注意：所有需要的变量现在都通过参数传递，以确保在 Windows 多进程中正常工作。
    """
    # 构建命令
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

    # Windows下调用子进程
    # 注意：这里不使用 shell=True，因为我们要直接调用 python
    exit_code = sp.run(cmd)

    if exit_code.returncode != 0:
        print(f"Error in subprocess for seed {seed}")
        return # 或者 sys.exit(1) 但在子进程中最好不要直接杀掉整个sys

    # 读取生成的 config 以前计算 loss
    try:
        config_path = f"{output_dir_path}/{wandb_name}/config.npy"
        if os.path.exists(config_path):
            config = np.load(config_path, allow_pickle=True)[()]
            loss_eval = np.array(config['loss_eval'])
            inds = np.argsort(loss_eval)
            # 将结果存入共享字典
            shared_losses[wandb_name] = loss_eval[inds][0]
        else:
            print(f"Config file not found: {config_path}")
    except Exception as e:
        print(f"Error reading config for {wandb_name}: {e}")


def display_(seed, wandb_name, output_dir_path, num_iter):
    # 此功能在纯命令行模式下可能无法正常显示，但在 Notebook 中有用
    save_interval = 10
    path_to_svg = f"{output_dir_path}/{wandb_name}/svg_logs/"
    intervals_ = list(range(0, num_iter, save_interval))
    
    # 简单的进度模拟
    print("Displaying progress...")
    for i in intervals_:
        filename = f"svg_iter{i}.svg"
        full_path = f"{path_to_svg}/{filename}"
        
        # 等待文件生成
        timeout = 0
        while not os.path.isfile(full_path):
            time.sleep(1)
            timeout += 1
            if timeout > 10: break # 防止死循环
        
        if os.path.isfile(full_path):
            # 这里原本是 Jupyter 的显示逻辑
            pass 

# -------------------------------------------------------------------------
# 主程序入口 (Windows 必须使用 if __name__ == "__main__":)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 路径与环境设置
    abs_path = os.path.abspath(os.getcwd())
    target = f"{abs_path}/target_images/{args.target_file}"
    
    assert os.path.isfile(target), f"{target} does not exists! Please check the path."

    # 2. 模型下载检查 (修复 Windows gdown 问题)
    model_path = f"{abs_path}/U2Net_/saved_models/u2net.pth"
    if not os.path.isfile(model_path):
        print(f"Downloading model to {model_path}...")
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Windows 必须加 shell=True 才能找到 gdown 命令
        try:
            sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                    "-O", "U2Net_/saved_models/u2net.pth"], shell=True)
        except Exception as e:
            print(f"Download failed: {e}. Please manually download u2net.pth")

    # 3. 输出目录设置
    test_name = os.path.splitext(args.target_file)[0]
    output_dir = f"{abs_path}/output_sketches/{test_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_iter = args.num_iter
    use_gpu = not args.cpu

    if not torch.cuda.is_available():
        use_gpu = False
        print("CUDA is not configured with GPU, running with CPU instead.")

    if args.colab:
        print("=" * 50)
        print(f"Processing [{args.target_file}] ...")
        if (args.colab or args.display) and Image_colab:
            try:
                img_ = Image_colab(target)
                display(img_)
            except: pass
        print(f"Results will be saved to \n[{output_dir}] ...")
        print("=" * 50)

    # 4. 多进程管理器初始化
    multiprocess = not args.colab and args.num_sketches > 1 and args.multiprocess
    seeds = list(range(0, args.num_sketches * 1000, 1000))

    # 使用 Manager 管理共享字典
    manager = mp.Manager()
    losses_all = manager.dict()

    # 5. 进程池执行
    if multiprocess:
        ncpus = min(10, mp.cpu_count())
        print(f"Starting multiprocessing pool with {ncpus} CPUS...")
        P = mp.Pool(ncpus)

    # 启动任务
    for seed in seeds:
        wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
        print(f"Launching task: {wandb_name}")
        
        if multiprocess:
            # 必须显式传递所有变量，不能依赖全局变量
            P.apply_async(run, args=(seed, wandb_name, target, output_dir, losses_all, args, use_gpu))
        else:
            run(seed, wandb_name, target, output_dir, losses_all, args, use_gpu)

    # 显示处理 (可选)
    if args.display and multiprocess:
        time.sleep(10)
        # P.apply_async(display_, ...) # Display 逻辑较复杂，建议在终端运行时忽略

    if multiprocess:
        P.close()
        P.join()  # 等待所有子进程结束

    # 6. 结果整理
    print("Finished processing. Selecting best sketch...")
    if len(losses_all) > 0:
        sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
        best_name = list(sorted_final.keys())[0]
        
        src_file = f"{output_dir}/{best_name}/best_iter.svg"
        dst_file = f"{output_dir}/{best_name}_best.svg"
        
        if os.path.exists(src_file):
            copyfile(src_file, dst_file)
            print(f"Best sketch saved to: {dst_file}")
        else:
            print("Best sketch file not found (maybe run failed).")
    else:
        print("No results found in losses_all. Check for errors above.")