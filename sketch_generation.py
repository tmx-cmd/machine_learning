import torch
import torch.nn.functional as F
import clip
import diffvg
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
import os

# -------- 配置 --------
prompt = "a sketch of a cat"
num_strokes = 48
canvas_size = (512, 512)
λ_sds, λ_clip, λ_reg = 1.0, 0.3, 0.01
steps = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------- 加载模型 --------
print("Loading models...")

# 加载 Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

# 加载 CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# CLIP 预处理（将 [0,1] 的 tensor 转换为 CLIP 输入）
clip_transform = transforms.Compose([
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
])

# -------- 工具函数 --------
def get_attention_map(pipe, prompt, canvas_size, num_inference_steps=20):
    """
    提取 Stable Diffusion 的 cross-attention map
    返回注意力图 (H, W)
    
    简化实现：生成一张图，然后基于图像内容创建注意力图
    更完整的实现需要使用 attention hook 提取真实的 cross-attention
    """
    # 方法1：生成一张参考图，然后基于图像创建注意力图
    with torch.no_grad():
        # 生成一张低分辨率图作为参考
        generator = torch.Generator(device=device).manual_seed(42)
        ref_image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # 转换为灰度图作为注意力图
        ref_image_gray = ref_image.convert('L')
        ref_array = np.array(ref_image_gray, dtype=np.float32) / 255.0
        
        # 调整大小到画布尺寸
        from PIL import Image
        ref_resized = Image.fromarray((ref_array * 255).astype(np.uint8))
        ref_resized = ref_resized.resize(canvas_size, Image.BILINEAR)
        attn_map = np.array(ref_resized, dtype=np.float32) / 255.0
        
        # 增强对比度，突出重要区域
        attn_map = np.power(attn_map, 0.7)
        
    return attn_map


def init_beziers_from_attention(attn_map, num_strokes, canvas_size, device):
    """
    根据注意力图初始化贝塞尔曲线
    返回: shapes, shape_groups, 可训练参数列表
    """
    H, W = canvas_size
    shapes = []
    shape_groups = []
    params = []
    
    # 确保注意力图是 numpy array
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.cpu().numpy()
    
    # 将注意力图归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 根据注意力强度采样点
    attn_flat = attn_map.flatten()
    probs = attn_flat / (attn_flat.sum() + 1e-8)
    
    # 采样高注意力区域
    try:
        indices = np.random.choice(len(attn_flat), size=min(num_strokes, len(attn_flat)), 
                                  p=probs, replace=False)
    except:
        # 如果采样失败，使用均匀采样
        indices = np.random.choice(len(attn_flat), size=min(num_strokes, len(attn_flat)), 
                                  replace=False)
    
    y_coords = indices // W
    x_coords = indices % W
    
    for i in range(len(indices)):
        # 每个笔画是一条贝塞尔曲线（3个点：起点、控制点、终点）
        x, y = float(x_coords[i]), float(y_coords[i])
        
        # 随机生成控制点（围绕采样点）
        offset = 15.0
        p0 = torch.tensor([x + random.uniform(-offset, offset), 
                          y + random.uniform(-offset, offset)], 
                         device=device, requires_grad=True, dtype=torch.float32)
        p1 = torch.tensor([x + random.uniform(-offset, offset), 
                          y + random.uniform(-offset, offset)], 
                         device=device, requires_grad=True, dtype=torch.float32)
        p2 = torch.tensor([x + random.uniform(-offset, offset), 
                          y + random.uniform(-offset, offset)], 
                         device=device, requires_grad=True, dtype=torch.float32)
        
        # 创建贝塞尔曲线路径（2个控制点 = 3个点）
        points = torch.stack([p0, p1, p2])
        num_control_points = torch.tensor([2], dtype=torch.int32)
        
        stroke_width = torch.tensor(2.0, device=device, requires_grad=True, dtype=torch.float32)
        
        path = diffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=stroke_width,
            is_closed=False,
            use_distance_approximation=True
        )
        
        shapes.append(path)
        
        # 创建形状组
        shape_group = diffvg.ShapeGroup(
            shape_ids=torch.tensor([i], dtype=torch.int32),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        )
        shape_groups.append(shape_group)
        
        # 收集可训练参数
        params.extend([p0, p1, p2, stroke_width])
    
    return shapes, shape_groups, params


def sds_loss(pipe, img_tensor, prompt, guidance_scale=7.5, t_range=[0.02, 0.98]):
    """
    Score Distillation Sampling (SDS) Loss
    img_tensor: [1, 3, H, W] in [0, 1]
    """
    # 调整图像大小到 VAE 输入尺寸 (512x512)
    if img_tensor.shape[-1] != 512 or img_tensor.shape[-2] != 512:
        img_tensor = F.interpolate(img_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    
    # 将图像编码到 latent space
    with torch.no_grad():
        # VAE encode
        img_normalized = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        posterior = pipe.vae.encode(img_normalized).latent_dist
        latents = posterior.sample()
        latents = latents * pipe.vae.config.scaling_factor
    
    # 添加噪声
    t = torch.randint(
        int(t_range[0] * pipe.scheduler.config.num_train_timesteps),
        int(t_range[1] * pipe.scheduler.config.num_train_timesteps),
        (1,),
        device=device
    ).long()
    
    noise = torch.randn_like(latents)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
    
    # 编码文本
    tokenizer = pipe.tokenizer
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    
    # UNet 预测噪声（需要梯度）
    unet = pipe.unet
    noise_pred = unet(
        noisy_latents,
        t,
        encoder_hidden_states=text_embeddings,
    ).sample
    
    # SDS Loss: 预测噪声和真实噪声的差异
    # 注意：这里 noise 是 detached 的，所以梯度只会流回 noise_pred，进而流回 latents
    loss = F.mse_loss(noise_pred, noise.detach())
    
    return loss


def clip_loss(clip_model, img_tensor, prompt, device):
    """
    CLIP 图文距离损失
    img_tensor: [1, 3, H, W] in [0, 1]
    """
    # 调整图像大小到 CLIP 输入尺寸 (224x224)
    img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
    # 归一化到 CLIP 的输入范围
    img_normalized = clip_transform(img_resized)
    
    # 编码图像和文本
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    image_features = clip_model.encode_image(img_normalized)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度损失
    similarity = (image_features @ text_features.t()).squeeze()
    loss = 1 - similarity
    
    return loss


def reg_loss(shapes):
    """
    正则化损失：鼓励平滑和合理的笔画长度
    """
    total_loss = torch.tensor(0.0, device=device)
    
    for shape in shapes:
        if isinstance(shape, diffvg.Path):
            points = shape.points
            if len(points) >= 2:
                # 鼓励控制点不要太分散
                if len(points) == 3:
                    # 对于贝塞尔曲线，计算控制点之间的距离
                    dist1 = torch.norm(points[1] - points[0])
                    dist2 = torch.norm(points[2] - points[1])
                    total_loss += 0.01 * (dist1 + dist2)
    
    return total_loss


# -------- 初始化 --------
print("Extracting attention map...")
attn_map = get_attention_map(pipe, prompt, canvas_size)

print("Initializing strokes from attention map...")
shapes, shape_groups, params = init_beziers_from_attention(
    attn_map, num_strokes, canvas_size, device
)

# 创建优化器
optimizer = torch.optim.Adam(params, lr=0.01)

print(f"Initialized {len(shapes)} strokes")
print("Starting optimization...")

# 创建输出目录
os.makedirs("outputs", exist_ok=True)

# -------- 训练循环 --------
for step in range(steps):
    optimizer.zero_grad()
    
    # 可微渲染
    scene_args = diffvg.RenderFunction.serialize_scene(
        canvas_size[0], canvas_size[1], shapes, shape_groups
    )
    
    img = diffvg.RenderFunction.apply(
        canvas_size[0], canvas_size[1], 2, 2, step, None, *scene_args
    )
    
    # 转换为 [1, 3, H, W] 格式，值范围 [0, 1]
    img_rgb = img[:, :, :3]  # 去掉 alpha
    img_tensor = img_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # 计算损失
    try:
        loss_sds = sds_loss(pipe, img_tensor, prompt)
        loss_clip = clip_loss(clip_model, img_tensor, prompt, device)
        loss_reg = reg_loss(shapes)
        
        loss = λ_sds * loss_sds + λ_clip * loss_clip + λ_reg * loss_reg
        
        loss.backward()
        
        # 梯度裁剪（防止不稳定）
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        
        optimizer.step()
        
        # 限制笔画宽度在合理范围
        for shape in shapes:
            if isinstance(shape, diffvg.Path):
                shape.stroke_width.data.clamp_(0.5, 5.0)
        
        if step % 20 == 0:
            print(f"[{step:3d}/{steps}] "
                  f"loss={loss.item():.4f} "
                  f"(SDS={loss_sds.item():.4f}, "
                  f"CLIP={loss_clip.item():.4f}, "
                  f"REG={loss_reg.item():.4f})")
            
            # 保存中间结果
            img_np = img_rgb.detach().cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(f"outputs/step_{step:03d}.png")
    
    except Exception as e:
        print(f"Error at step {step}: {e}")
        continue

print("Optimization complete!")

# -------- 导出最终结果 --------
print("Saving final SVG...")
diffvg.save_svg(
    "outputs/cat_sketch.svg",
    canvas_size[0],
    canvas_size[1],
    shapes,
    shape_groups
)

# 保存最终渲染图
with torch.no_grad():
    scene_args_final = diffvg.RenderFunction.serialize_scene(
        canvas_size[0], canvas_size[1], shapes, shape_groups
    )
    img_final = diffvg.RenderFunction.apply(
        canvas_size[0], canvas_size[1], 2, 2, steps, None, *scene_args_final
    )
    img_final_rgb = img_final[:, :, :3].detach().cpu().numpy()
    img_final_pil = Image.fromarray((img_final_rgb * 255).astype(np.uint8))
    img_final_pil.save("outputs/cat_sketch_final.png")

print("Done! Check 'outputs/cat_sketch.svg' and 'outputs/cat_sketch_final.png'")
