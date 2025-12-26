import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
import os
import math

# -------- 检查依赖 --------
try:
    import cv2
    print("Success: Imported 'cv2'.")
except ImportError:
    print("Error: OpenCV not found. Please run 'pip install opencv-python'")
    exit()

try:
    import pydiffvg
    import diffvg
    print("Success: Imported 'pydiffvg'.")
    USE_PYDIFFVG = True
except ImportError:
    import diffvg
    print("Warning: Using raw 'diffvg' (Hard Mode).")
    USE_PYDIFFVG = False

# -------- 1. 参数配置 (关键修改) --------
PROMPT = "one single continuous line drawing of a cat, vector art, minimalist, white background, black thick lines, no shading"
NEGATIVE_PROMPT = "sketch, pencil, texture, shading, noise, complex, detailed, realistic, gray, filling, messy, dots"
# 这里的 NUM_STROKES 不再硬性限制，而是根据轮廓自动生成
CANVAS_SIZE = (512, 512)
STEPS = 300             # 减少步数，因为初始化已经很好了，不需要动太多
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------- 2. 模型加载 --------
print("Loading Stable Diffusion...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
except Exception as e:
    print(f"Error: {e}")
    exit()

print("Loading CLIP...")
import clip
try:
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
except:
    clip_model = None

# -------- 3. 生成参考图 --------
print(f"Generating reference...")
os.makedirs("outputs", exist_ok=True)
with torch.no_grad():
    target_pil = pipe(PROMPT, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=30, guidance_scale=9.0).images[0]
target_pil = ImageOps.autocontrast(target_pil, cutoff=5)
target_pil.save("outputs/target_reference.png")

target_tensor = transforms.ToTensor()(target_pil).to(DEVICE).unsqueeze(0)
target_gray_np = np.array(target_pil.convert("L"))

# -------- 4. 核心：切分轮廓初始化 --------

def add_stroke(p0, p1, p2, width_val, shapes, shape_groups, p_vars, w_vars):
    # 转换为 Tensor
    points = torch.tensor([p0, p1, p2], dtype=torch.float32, device=DEVICE).contiguous()
    points.requires_grad = True # 开启优化
    p_vars.append(points)
    
    width = torch.tensor(width_val, dtype=torch.float32, device=DEVICE).contiguous()
    width.requires_grad = True
    w_vars.append(width)
    
    if USE_PYDIFFVG:
        path = pydiffvg.Path(
            num_control_points=torch.tensor([1], dtype=torch.int32),
            points=points,
            stroke_width=width,
            is_closed=False
        )
        shapes.append(path)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes)-1], dtype=torch.int32),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE)
        )
        shape_groups.append(group)
    else:
        # Fallback
        path = diffvg.Path(
            torch.tensor([1], dtype=torch.int32).cpu(), points, width.unsqueeze(0), 0, 0, False, True
        )
        shapes.append(path)
        group = diffvg.ShapeGroup(
            torch.tensor([len(shapes)-1], dtype=torch.int32).cpu(), None, 
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE)
        )
        shape_groups.append(group)

def init_shapes_smart(target_img_np, canvas_size):
    print("Using Smart Contour Segmentation...")
    H, W = canvas_size
    
    # 1. 预处理：二值化
    _, binary = cv2.threshold(target_img_np, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    shape_groups = []
    points_vars = []
    width_vars = []
    
    # 3. 遍历轮廓并切分
    for cnt in contours:
        # 忽略太短的噪点
        if cv2.arcLength(cnt, False) < 20:
            continue
            
        # 简化轮廓，减少点数
        epsilon = 0.005 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        
        # 将轮廓点提取为列表
        pts = approx.squeeze().tolist()
        if len(pts) < 2: continue
        
        # 如果是封闭的，把终点加回去
        # if cv2.isContourConvex(approx):
        #     pts.append(pts[0])
            
        # 4. 关键：将长轮廓切分为多个贝塞尔曲线段
        # 每两个点之间插一个贝塞尔曲线
        for i in range(len(pts) - 1):
            p_start = pts[i]
            p_end = pts[i+1]
            
            # 简单的取中点作为初始控制点
            p_mid = [(p_start[0] + p_end[0])/2, (p_start[1] + p_end[1])/2]
            
            # 加入一点点随机扰动，防止梯度死锁
            p_mid[0] += random.uniform(-1, 1)
            p_mid[1] += random.uniform(-1, 1)
            
            add_stroke(p_start, p_mid, p_end, 2.5, shapes, shape_groups, points_vars, width_vars)
            
    print(f"Initialized {len(shapes)} segments from contours.")
    if len(shapes) == 0:
        print("Warning: No shapes found! Adding a dummy stroke.")
        add_stroke([256,256], [260,260], [270,270], 2.0, shapes, shape_groups, points_vars, width_vars)
        
    return shapes, shape_groups, points_vars, width_vars

print("Initializing strokes...")
shapes, shape_groups, points_vars, width_vars = init_shapes_smart(target_gray_np, CANVAS_SIZE)

background = torch.ones((CANVAS_SIZE[1], CANVAS_SIZE[0], 4), dtype=torch.float32, device=DEVICE)
background.requires_grad = False 

# -------- 5. 优化循环 --------
optimizer = torch.optim.Adam(points_vars + width_vars, lr=0.5) 
print("Starting optimization...")

for step in range(STEPS):
    optimizer.zero_grad()
    
    if USE_PYDIFFVG:
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups
        )
        render = pydiffvg.RenderFunction.apply
    else:
        scene_args = diffvg.RenderFunction.serialize_scene(
            CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups
        )
        render = diffvg.RenderFunction.apply
    
    img = render(
        CANVAS_SIZE[0], CANVAS_SIZE[1], 2, 2, step, background, *scene_args
    )
    
    img_tensor = img[:, :, :3].permute(2, 0, 1).unsqueeze(0)
    
    # Loss 1: MSE (像素) - 权重极大，确保拟合
    loss_mse = F.mse_loss(img_tensor, target_tensor)
    
    # Loss 2: Reg (正则化) - ！！关键修改！！
    # 既然之前塌缩成点了，我们这次直接设为 0，或者非常非常小
    loss_reg = torch.tensor(0.0, device=DEVICE)
    # 仅当 step 后期才稍微加一点点平滑约束，前期完全不加
    
    # Loss 3: CLIP
    loss_clip = torch.tensor(0.0, device=DEVICE)
    if clip_model is not None and step % 50 == 0:
        img_sm = F.interpolate(img_tensor, (224, 224), mode='bilinear')
        norm = transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))(img_sm)
        img_feat = clip_model.encode_image(norm)
        txt_feat = clip_model.encode_text(clip.tokenize(PROMPT).to(DEVICE))
        loss_clip = 1.0 - (img_feat / img_feat.norm(dim=-1, keepdim=True) @ 
                          (txt_feat / txt_feat.norm(dim=-1, keepdim=True)).t()).squeeze()

    # 总 Loss：只关注 MSE，不要 Reg
    loss = loss_mse * 2000.0 + loss_clip * 1.0
    loss.backward()
    
    if USE_PYDIFFVG:
        for path, p_var, w_var in zip(shapes, points_vars, width_vars):
            path.points = p_var
            path.stroke_width = w_var
            
    optimizer.step()
    
    for w in width_vars: w.data.clamp_(1.0, 5.0)
        
    if step % 50 == 0:
        print(f"Step {step}/{STEPS}: Loss={loss.item():.4f} (MSE={loss_mse.item()*2000:.4f})")

# -------- 6. 保存 --------
print("Saving...")
if USE_PYDIFFVG:
    pydiffvg.save_svg("outputs/cat_final.svg", CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups)
else:
    diffvg.save_svg("outputs/cat_final.svg", CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups)

with torch.no_grad():
    if USE_PYDIFFVG:
        scene_args = pydiffvg.RenderFunction.serialize_scene(CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups)
        img = pydiffvg.RenderFunction.apply(CANVAS_SIZE[0], CANVAS_SIZE[1], 2, 2, STEPS, background, *scene_args)
    else:
        scene_args = diffvg.RenderFunction.serialize_scene(CANVAS_SIZE[0], CANVAS_SIZE[1], shapes, shape_groups)
        img = diffvg.RenderFunction.apply(CANVAS_SIZE[0], CANVAS_SIZE[1], 2, 2, STEPS, background, *scene_args)
        
    Image.fromarray((img.cpu().numpy()*255).astype(np.uint8)).save("outputs/cat_final_render.png")

print("Done.")