import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os
import glob
import numpy as np
import yaml  # 新增：用于解析配置文件
import argparse

# 导入你的架构
from xreflection.archs.dsrnet_arch import DSRNet, QuantArgs
from xreflection.archs.quantize_arch import QuantConv2d


# ==========================================
# VGG Extractor (保持不变)
# ==========================================
class VGGExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30): self.slice5.add_module(str(x), vgg[x])
        self.to(device)
        self.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


# ==========================================
# Dataset (保持不变)
# ==========================================
class CalibrationDataset(Dataset):
    def __init__(self, img_dir, size=224):
        super().__init__()
        # 支持递归查找图片
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True) +
                                glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True))

        # 限制数量，校准不需要太多图片，50-100张足矣
        if len(self.img_paths) > 100:
            # 随机采样而不是只取前100张，以获得更广泛的分布
            np.random.seed(42)
            self.img_paths = np.random.choice(self.img_paths, 100, replace=False).tolist()

        print(f"Found {len(self.img_paths)} images for calibration in {img_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)


# ==========================================
# 核心校准逻辑 (支持 YAML 配置)
# ==========================================
def calibrate_model(config_path, fp32_ckpt, save_path, device='cuda'):
    print(f"=== Starting Calibration (Config-Driven) ===")

    # 1. 加载 YAML 配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    net_cfg = config['network_g']

    # 2. [修改] 直接使用导入的 QuantArgs 类
    # 利用 **kwargs 特性，直接传入校准专用的参数 (如 layerwise)
    # 这样既使用了默认值，又覆盖了校准所需的特殊配置
    args = QuantArgs(
        quantize=True,
        quantize_a=8.0,
        quantize_w=8.0,
        quantizer='percentile',
        quantizer_w='omse',
        percentile_alpha=0.999,
        ema_beta=0.99,
        wbc=True,
        # --- AdaBM 校准特有参数 ---
        layerwise=True,  # 校准时开启 layerwise 统计
        imgwise=False,
        layer_percentile=0.1  # 统计截断阈值
    )

    # 3. 实例化模型
    # 注意：这里我们使用新的统一接口 quant_config
    model = DSRNet(
        width=net_cfg['width'],
        middle_blk_num=net_cfg['middle_blk_num'],
        enc_blk_nums=net_cfg['enc_blk_nums'],
        dec_blk_nums=net_cfg['dec_blk_nums'],
        lrm_blk_nums=net_cfg['lrm_blk_nums'],
        quant_config=args  # <--- 传入 args 对象，参数名改为 quant_config (如果你已经按上一条建议修改了 DSRNet)
        # 如果还没修改 DSRNet 接口，依然用 args=args
    )

    # 4. 加载权重
    print(f"Loading FP32 model from {fp32_ckpt}...")
    try:
        checkpoint = torch.load(fp32_ckpt, map_location='cpu')
        # 处理 LightningCheckpoint 的嵌套结构
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Lightning 可能会给参数名加前缀 "net_g." 或 "model."，需要去除
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net_g.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # 仅用于调试：如果没有权重，也可以继续运行看流程是否通
        # return

    model.to(device)
    model.eval()

    # 5. 准备数据 (从 YAML 的验证集读取)
    # 尝试读取第一个验证集路径
    try:
        val_dataset_cfg = config['datasets']['val_datasets'][0]
        data_dir = val_dataset_cfg['datadir']
        print(f"Using validation dataset from config: {val_dataset_cfg['name']} -> {data_dir}")
    except:
        print("Warning: Could not read validation path from config. Please specify manually if needed.")
        data_dir = 'datasets/test/mixed'  # Fallback

    # 获取输入尺寸，优先使用训练配置的尺寸
    calib_size = 224
    if 'train' in config['datasets'] and 'fused_datasets' in config['datasets']['train']:
        # 尝试读取第一个 fused dataset 的 transform_size
        try:
            calib_size = config['datasets']['train']['fused_datasets'][0]['transform_size']
            print(f"Using calibration image size: {calib_size} (from config)")
        except:
            pass

    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Using Random Data for testing flow.")
        use_random = True
        dataloader = range(10)
    else:
        dataset = CalibrationDataset(data_dir, size=calib_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        vgg_extractor = VGGExtractor(device)
        use_random = False

    # 6. 初始化 QuantConv2d
    print("Enabling initialization mode...")
    for m in model.modules():
        if m.__class__.__name__ == 'QuantConv2d':
            m.init_mode = True
            m.ema_epoch = 0
            m.w_bit = 32.0
            m.a_bit = 32.0

    # 7. 运行校准
    print("Running forward passes to collect statistics...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if use_random:
                inp = torch.randn(1, 3, calib_size, calib_size).to(device)
                H, W = calib_size, calib_size
                feats = [torch.randn(1, 64, H, W).to(device) for _ in range(5)]
            else:
                inp = batch.to(device)
                feats = vgg_extractor(inp)

            model(inp, feats)

            if i % 10 == 0: print(f"  - Batch {i} processed.")
            if i >= 50: break  # 50张图片足够校准

    # 8. Layerwise 统计逻辑 (AdaBM)
    if args.layerwise:
        print("Performing Layerwise Calibration stats check...")
        range_list = []
        for m in model.modules():
            if m.__class__.__name__ == 'QuantConv2d':
                current_range = (m.upper_a - m.lower_a).abs().mean().item()
                range_list.append(current_range)

        lower_bound = np.percentile(range_list, args.layer_percentile)
        upper_bound = np.percentile(range_list, 100.0 - args.layer_percentile)
        print(
            f"  Global Range Stats: Lower(p{args.layer_percentile})={lower_bound:.4f}, Upper(p{100 - args.layer_percentile})={upper_bound:.4f}")

    # 9. 保存
    print("Freezing parameters and saving...")
    for m in model.modules():
        if m.__class__.__name__ == 'QuantConv2d':
            m.init_mode = False
            m.w_bit = args.quantize_w
            m.a_bit = args.quantize_a

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"=== Quantized model saved to {save_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../options/train_dsrnet.yml', help='Path to training config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained FP32 checkpoint')
    parser.add_argument('--save_path', type=str, default='experiments/dsrnet_quant_calib_percentile.pth', help='Output path')
    args = parser.parse_args()

    calibrate_model(args.config, args.ckpt, args.save_path)
