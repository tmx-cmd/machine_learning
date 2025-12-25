# 依赖安装指南

## 1. 基础环境（推荐用 conda）

```bash
# 创建 conda 环境
conda create -n sketch_gen python=3.9
conda activate sketch_gen

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 或 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 2. diffvg（必须 git clone + 编译）

```bash
# 克隆仓库
git clone https://github.com/BachiLi/diffvg.git
cd diffvg

# 安装依赖
pip install svgpathtools scikit-image

# 编译安装（Windows 需要 Visual Studio Build Tools）
git submodule update --init --recursive
python setup.py install

# 如果编译失败，参考：https://github.com/BachiLi/diffvg#installation
```

**注意**：Windows 上编译 diffvg 可能需要：
- Visual Studio Build Tools（C++ 编译器）
- CMake
- 可能需要修改 setup.py 中的编译选项

## 3. CLIP（pip 安装）

```bash
pip install git+https://github.com/openai/CLIP.git
# 或
pip install clip-by-openai
```

## 4. Stable Diffusion（pip 安装）

```bash
# 推荐使用 diffusers（更简单）
pip install diffusers transformers accelerate

# 如果需要 cross-attention 提取，可能需要额外安装
pip install xformers  # 可选，用于优化注意力计算
```

## 5. 其他依赖

```bash
pip install pillow numpy scipy
```

## 快速安装脚本（Linux/Mac）

```bash
conda create -n sketch_gen python=3.9 -y
conda activate sketch_gen
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
pip install svgpathtools scikit-image
git submodule update --init --recursive
python setup.py install
cd ..

pip install git+https://github.com/openai/CLIP.git
pip install diffusers transformers accelerate pillow numpy scipy
```

## Windows 特殊说明

1. **diffvg 编译**：需要安装 Visual Studio Build Tools（包含 C++ 编译器）
2. **如果编译困难**：可以考虑使用 WSL2（Windows Subsystem for Linux）
3. **替代方案**：如果 diffvg 编译失败，可以考虑使用其他可微渲染库，但需要修改代码

## 验证安装

创建 `test_imports.py`：

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import diffvg
    print("✓ diffvg installed")
except ImportError as e:
    print(f"✗ diffvg not installed: {e}")

try:
    import clip
    print("✓ CLIP installed")
except ImportError as e:
    print(f"✗ CLIP not installed: {e}")

try:
    from diffusers import StableDiffusionPipeline
    print("✓ diffusers installed")
except ImportError as e:
    print(f"✗ diffusers not installed: {e}")
```

运行：`python test_imports.py`

