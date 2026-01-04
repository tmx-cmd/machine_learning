# Text2Sketch: 基于CLIPasso的文本转素描系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于CLIPasso开发的创新文本转素描系统，通过结合Stable Diffusion和CLIPasso技术，将文本描述直接转换为高质量的矢量素描。

## 🎯 项目特色

- **纯提示词工程**: 无需rembg等外部依赖，完全依赖精心设计的提示词
- **多风格支持**: 支持写实、动漫、涂鸦等多种艺术风格
- **复杂度控制**: 三级复杂度设置，从简单图标到复杂场景
- **自动化评估**: 内置基准测试和质量评估系统
- **消融实验**: 支持对比不同方法的生成效果

## 📋 核心功能

### 1. 文本转素描生成

###英文prompt效果好一点
```python
from clipasso_api import text_to_image

result = text_to_image(
    prompt="一只可爱的小猫",
    complexity="medium",
    style="realistic"
)
```

### 2. 复杂度控制
- **low** (16笔画): 简单图标和图形
- **medium** (32笔画): 中等复杂度对象
- **high** (48笔画): 复杂场景和细节

### 3. 风格选项
- **default**: 标准矢量风格
- **anime**: 动漫风格，适合卡通化
- **realistic**: 写实风格，更高的细节度
- **scribble**: 涂鸦风格，手绘感

### 4. 基准测试
内置SketchBench-10数据集，支持自动化质量评估，包括：
- 客观指标：笔画数一致性
- 主观指标：语义和美学评分（基于Qwen3-VL）

## 🚀 快速开始

### 环境要求
- Python 3.7+
- CUDA-compatible GPU (推荐，用于加速生成)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/text2sketch.git
cd text2sketch
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **下载CLIPasso模型**
```bash
# 系统会自动下载U2Net模型到CLIPasso-main/U2Net_/saved_models/
python clipasso_api.py  # 首次运行会自动下载必要模型
```

## 📖 使用指南

### 基本使用

```python
from clipasso_api import text_to_image

# 生成写实风格的猫咪素描
result = text_to_image(
    prompt="一只白色波斯猫坐在椅子上",
    complexity="high",
    style="realistic",
    output_dir="./output"
)

if result["success"]:
    print(f"基础图像: {result['base_image_path']}")
    print(f"素描结果: {result['best_sketch_path']}")
```

### 高级参数

```python
result = text_to_image(
    prompt="魔法少女手持宝剑",
    negative_prompt="模糊，低质量，变形",  # 负面提示词
    complexity="medium",  # 复杂度：low/medium/high
    style="anime",        # 风格：default/anime/realistic/scribble
    output_dir="./sketches",
    use_gpu=True,         # 是否使用GPU
    multiprocess=False    # 是否启用多进程
)
```

### 消融实验

运行消融实验比较不同方法的生成效果：

```bash
python clipasso_example.py ablation
```

这将生成三个实验组的结果：
1. 直接Stable Diffusion生成
2. CLIPasso无提示词工程
3. CLIPasso有提示词工程

### 基准测试

运行完整的基准测试评估：

```python
from benchmark_eval import run_benchmark_evaluation

# 运行SketchBench-10评估
run_benchmark_evaluation()
```

## 📊 技术架构

```
文本提示词 → Stable Diffusion → 基础图像 → CLIPasso → 矢量素描
     ↓              ↓              ↓              ↓
  提示词工程    白底强制      显著性检测      SVG优化
```

### 核心组件

1. **提示词工程模块** (`process_prompt_engineering`)
   - 强制白底、无阴影
   - 居中构图、完整主体
   - 风格化增强

2. **Stable Diffusion集成**
   - runwayml/stable-diffusion-v1-5模型
   - DPM++采样器优化
   - 负面提示词强化

3. **CLIPasso适配器**
   - 自动路径检测
   - 多进程支持
   - 损失函数优化

4. **评估系统**
   - SVG笔画数统计
   - 大模型质量评分
   - 自动化报告生成


## 🔧 配置说明

### GPU支持
- 自动检测CUDA可用性
- CPU回退模式
- 显存优化（attention slicing）

### 路径配置
- 自动检测CLIPasso安装路径
- 支持自定义输出目录
- 临时文件自动清理

## 📈 性能优化

- **生成时间**: 低复杂度 ~2-3分钟，中等 ~4-5分钟，高复杂度 ~6-8分钟
- **GPU加速**: 推荐使用NVIDIA GPU，生成速度提升3-5倍
- **多进程**: 支持并发生成多个素描
- **内存优化**: 自动启用attention slicing降低显存占用



### 开发环境设置
```bash
# 创建虚拟环境
conda create -n text2sketch python=3.8
conda activate text2sketch

# 安装开发依赖
pip install -r requirements-dev.txt
```



## 🙏 致谢

- [CLIPasso](https://github.com/) - 核心素描生成算法
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 基础图像生成
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 模型集成



---

**⭐ 如果这个项目对你有帮助，请给我们一个star！**
