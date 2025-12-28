# CLIPasso API Wrapper

这是一个CLIPasso的API包装器，将CLIPasso-main的功能封装成一个易于调用的Python函数接口。

## 文件结构

```
E:\mllab\machine_learning\
├── CLIPasso-main\           # 原始CLIPasso代码（未修改）
├── clipasso_api.py          # API包装器主文件
├── clipasso_example.py      # 使用示例
└── README_CLIPasso_API.md   # 本说明文档
```

## 功能特性

- 🚀 **简单易用**: 单函数调用即可生成草图或文生图
- 🎨 **文生图**: 支持文本到图像生成（Stable Diffusion + CLIPasso）
- ⚡ **灵活配置**: 支持所有CLIPasso参数定制
- 🔄 **多进程支持**: 自动并行处理多个草图生成
- 🎯 **智能选择**: 自动选择质量最好的草图
- 💾 **自定义输出**: 指定输出目录和文件名

## 安装要求

1. 确保CLIPasso-main文件夹位于同一目录下
2. 安装必要的依赖：
   ```bash
   # 基础CLIPasso依赖（在CLIPasso-main目录中运行）
   pip install -r requirements.txt

   # 文生图功能额外依赖
   pip install diffusers torch accelerate transformers
   ```
3. 确保有足够的磁盘空间用于模型下载和输出（Stable Diffusion模型约4GB）

## 快速开始

### 基本用法

#### 图生图（图像到素描）
```python
from clipasso_api import generate_sketch

result = generate_sketch(
    target_file="path/to/your/image.jpg",
    num_strokes=16,
    num_iter=1000
)

if result["success"]:
    print(f"草图生成成功: {result['best_sketch_path']}")
```

#### 文生图（文本到素描）
```python
from clipasso_api import text_to_image

result = text_to_image(
    prompt="一只可爱的小猫在花园里玩耍",
    negative_prompt="模糊，低质量",
    num_strokes=32,
    num_iter=1500
)

if result["success"]:
    print(f"素描生成成功: {result['best_sketch_path']}")
```

### 高级用法

```python
result = generate_sketch(
    target_file="path/to/your/image.jpg",
    num_strokes=32,        # 更多笔画 = 更精细
    num_iter=2001,         # 更多迭代 = 更好质量
    fix_scale=1,           # 固定缩放非正方形图片
    mask_object=1,         # 遮罩背景
    num_sketches=3,        # 生成3个草图并选择最佳
    use_gpu=True,          # 强制使用GPU
    output_dir="my_output"  # 自定义输出目录
)
```

## API参考

### `text_to_image()` 函数

文生图函数：先用Stable Diffusion生成图像，再用CLIPasso生成素描。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | 必需 | 正向文本提示词 |
| `negative_prompt` | str | "" | 负向提示词 |
| `output_dir` | str | "./generated_images" | 输出目录 |
| `filename` | str | None | 输出文件名（None=自动生成） |
| `num_strokes` | int | 16 | 素描笔画数量 |
| `num_iter` | int | 2001 | CLIPasso迭代次数 |
| `fix_scale` | int | 0 | 是否固定缩放非正方形图片 |
| `mask_object` | int | 0 | 是否遮罩背景 |
| `num_sketches` | int | 3 | 生成素描数量 |
| `use_gpu` | bool | None | 是否使用GPU（None=自动检测） |
| `clipasso_path` | str | None | CLIPasso-main路径（None=自动检测） |
| `multiprocess` | bool | True | 是否使用多进程 |

#### 返回值

```python
{
    "success": bool,           # 是否成功
    "prompt": str,             # 使用的提示词
    "negative_prompt": str,    # 使用的负向提示词
    "base_image_temp_path": str, # 临时基础图像路径
    "output_dir": str,         # 输出目录路径
    "best_sketch_path": str,   # 最佳素描文件路径
    "all_sketches": list,      # 所有生成的素描路径列表
    "losses": dict,           # 各素描的损失值字典
    "sketch_result": dict,    # 完整的CLIPasso结果
    "error": str              # 错误信息（如果有）
}
```

### `generate_sketch()` 函数

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_file` | str | 必需 | 目标图片文件路径 |
| `num_strokes` | int | 16 | 草图笔画数量 |
| `num_iter` | int | 2001 | 迭代次数 |
| `fix_scale` | int | 0 | 是否固定缩放非正方形图片 |
| `mask_object` | int | 0 | 是否遮罩背景 |
| `num_sketches` | int | 3 | 生成草图数量 |
| `use_gpu` | bool | None | 是否使用GPU（None=自动检测） |
| `output_dir` | str | None | 输出目录（None=自动生成） |
| `clipasso_path` | str | None | CLIPasso-main路径（None=自动检测） |
| `multiprocess` | bool | True | 是否使用多进程 |

#### 返回值

```python
{
    "success": bool,           # 是否成功
    "output_dir": str,         # 输出目录路径
    "best_sketch_path": str,   # 最佳草图文件路径
    "all_sketches": list,      # 所有生成的草图路径列表
    "losses": dict,           # 各草图的损失值字典
    "error": str              # 错误信息（如果有）
}
```

## 使用示例

### 1. 文生图基础用法

```python
from clipasso_api import text_to_image

result = text_to_image(
    prompt="一只可爱的小猫",
    negative_prompt="模糊，低质量，变形"
)
if result["success"]:
    print(f"素描生成成功: {result['best_sketch_path']}")
```

### 2. 文生图高级用法

```python
result = text_to_image(
    prompt="美丽的山水画风格风景",
    negative_prompt="现代建筑，城市，人物",
    num_strokes=48,        # 更多笔画 = 更精细
    num_iter=2001,         # 完整迭代次数
    fix_scale=1,           # 固定缩放
    mask_object=0,         # 不遮罩背景
    num_sketches=2,        # 生成2个素描
    output_dir="./my_art"  # 自定义输出目录
)
```

### 3. 图生图基础用法

```python
from clipasso_api import generate_sketch

result = generate_sketch("cat.jpg")
print(result["best_sketch_path"])  # 输出最佳草图路径
```

### 2. 批量处理多图

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

for img in images:
    result = generate_sketch(img, num_strokes=24, num_iter=1500)
    if result["success"]:
        print(f"✓ {img} -> {result['best_sketch_path']}")
```

### 3. 自定义参数精细控制

```python
result = generate_sketch(
    target_file="portrait.jpg",
    num_strokes=64,      # 高细节
    num_iter=5000,       # 高质量
    fix_scale=1,         # 保持比例
    mask_object=1,       # 移除背景
    num_sketches=5       # 多候选选择
)
```

## 输出文件结构

```
output_directory/
├── sketch_name_16strokes_seed0/
│   ├── best_iter.svg          # 单个草图结果
│   ├── config.npy            # 配置和损失数据
│   └── svg_logs/             # 迭代过程日志
├── sketch_name_16strokes_seed1000/
│   └── ...                   # 其他种子结果
└── sketch_name_16strokes_seed0_best.svg  # 最佳草图副本
```

## 故障排除

### 常见问题

1. **找不到CLIPasso路径**
   - 确保CLIPasso-main文件夹与clipasso_api.py在同一目录

2. **模型下载失败**
   - 手动下载u2net.pth到`CLIPasso-main/U2Net_/saved_models/`

3. **CUDA不可用**
   - 函数会自动回退到CPU模式

4. **生成失败**
   - 检查目标图片是否存在且为有效格式
   - 确认有足够磁盘空间

5. **文生图功能无法使用**
   - 确认已安装diffusers: `pip install diffusers torch accelerate`
   - 检查网络连接（首次运行需要下载Stable Diffusion模型）

6. **Stable Diffusion模型下载慢**
   - 使用镜像源或VPN加速下载
   - 模型大小约4GB，下载时间较长

### 调试模式

启用详细输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

- **GPU加速**: 确保CUDA可用
- **多进程**: 对于多个草图，启用`multiprocess=True`
- **批处理**: 减少`num_iter`以加快处理
- **内存管理**: 避免同时处理太多大图片

## 许可证

遵循原始CLIPasso项目的许可证。
