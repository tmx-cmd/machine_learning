# DiffSketcher: 基于 Diffusion + CLIP 的矢量素描生成

## 项目简介

这个项目实现了"强强联合"的矢量素描生成方法：
- **Generator (Diffusion)**: 提供先验知识和注意力图
- **Evaluator (CLIP)**: 确保语义一致性
- **Optimizer**: 优化贝塞尔曲线参数（而非像素）

## 核心流程

1. **第一阶段：获取先验**
   - 使用 Stable Diffusion 生成参考图
   - 提取注意力图，确定重要区域

2. **第二阶段：联合优化**
   - 在注意力热区初始化贝塞尔曲线
   - 使用 diffvg 进行可微渲染
   - 通过 SDS Loss + CLIP Loss 优化曲线参数
   - 迭代 200 步，自动收敛到最佳形状

## 安装依赖

请参考 `INSTALL.md` 文件，主要步骤：

1. 创建 conda 环境并安装 PyTorch
2. 编译安装 diffvg（需要 git clone）
3. 安装其他依赖（pip）

## 使用方法

```bash
# 激活环境
conda activate sketch_gen

# 运行代码
python sketch_generation.py
```

## 配置参数

在 `sketch_generation.py` 中可以调整：

- `prompt`: 文本提示词（默认："a sketch of a cat"）
- `num_strokes`: 笔画数量（默认：48）
- `canvas_size`: 画布大小（默认：(512, 512)）
- `steps`: 优化步数（默认：200）
- `λ_sds, λ_clip, λ_reg`: 损失权重（默认：1.0, 0.3, 0.01）

## 输出文件

- `outputs/cat_sketch.svg`: 最终矢量图（SVG 格式）
- `outputs/cat_sketch_final.png`: 最终渲染图（PNG 格式）
- `outputs/step_XXX.png`: 中间步骤的渲染结果

## 注意事项

1. **首次运行**：会下载 Stable Diffusion 模型（约 4GB），需要网络连接
2. **GPU 推荐**：建议使用 CUDA GPU，CPU 运行会很慢
3. **内存需求**：至少需要 8GB GPU 内存
4. **Windows 用户**：diffvg 编译可能需要 Visual Studio Build Tools

## 故障排除

### diffvg 编译失败
- Windows: 安装 Visual Studio Build Tools
- Linux/Mac: 确保安装了 GCC 和 CMake
- 备选方案: 使用 WSL2（Windows）

### CUDA 内存不足
- 减小 `canvas_size`（如改为 256x256）
- 减少 `num_strokes`
- 使用 CPU（会很慢）

### 模型下载失败
- 检查网络连接
- 可能需要配置代理或使用镜像源

## 代码结构

- `sketch_generation.py`: 主程序
- `get_attention_map()`: 提取注意力图
- `init_beziers_from_attention()`: 初始化笔画
- `sds_loss()`: Score Distillation Sampling 损失
- `clip_loss()`: CLIP 语义损失
- `reg_loss()`: 正则化损失

## 参考

- diffvg: https://github.com/BachiLi/diffvg
- Stable Diffusion: https://github.com/runwayml/stable-diffusion
- CLIP: https://github.com/openai/CLIP

