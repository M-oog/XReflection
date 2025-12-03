import torch
import sys
import os

# 确保能导入项目模块 (假设你在项目根目录运行)
sys.path.append(os.getcwd())

from xreflection.archs.dsrnet_arch import DSRNet, QuantArgs


def check_keys():
    # 1. 实例化模型 (FP32 模式)
    # 确保这里的参数与 test_dsrnet.yml 中的完全一致
    args = QuantArgs(quantize=False)
    model = DSRNet(
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        lrm_blk_nums=[2, 4],
        quant_config=args
    )
    model_keys = set(model.state_dict().keys())

    # 2. 加载权重文件
    ckpt_path = '/mnt/hugedisk/mxt/projects/XReflection/pretrained_model/dsr-25.8915.ckpt'
    print(f"Checking checkpoint: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 提取参数字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Type: Lightning Checkpoint (state_dict detected)")
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
        print("Type: Custom Checkpoint (params detected)")
    else:
        state_dict = checkpoint
        print("Type: Raw State Dict")

    ckpt_keys = set(state_dict.keys())

    # 3. 对比分析
    print(f"\nModel keys count: {len(model_keys)}")
    print(f"Ckpt keys count:  {len(ckpt_keys)}")

    matched = 0
    missing_in_ckpt = []  # 记录在 checkpoint 中找不到的模型参数

    # 遍历模型需要的每一个 key，去 checkpoint 里找
    for m_key in model_keys:
        # VGG 和 num_batches_tracked 不需要检查
        if "vgg" in m_key or "num_batches_tracked" in m_key:
            continue

        # 尝试匹配多种前缀情况
        candidates = [
            m_key,  # 情况1: 完全匹配 (intro.weight)
            f"net_g.{m_key}",  # 情况2: Ckpt带前缀 (net_g.intro.weight)
            f"module.{m_key}",  # 情况3: DDP带前缀 (module.intro.weight)
            f"model.{m_key}",  # 情况4: 通用前缀
            m_key.replace("module.", "")  # 情况5: 模型带前缀但ckpt没带
        ]

        found = False
        for c in candidates:
            if c in ckpt_keys:
                found = True
                break

        if found:
            matched += 1
        else:
            missing_in_ckpt.append(m_key)

    print(f"Matched DSRNet keys: {matched}")
    print(f"Missing DSRNet keys: {len(missing_in_ckpt)}")

    if len(missing_in_ckpt) > 0:
        print("\n--- Top 20 Missing Keys in Checkpoint ---")
        # [修正] 这里的变量名之前写错了
        for k in sorted(missing_in_ckpt)[:20]:
            print(f"  {k}")
    else:
        print("\nSUCCESS: All critical DSRNet keys are found!")

    # 4. 特别检查融合参数 (Fusion Parameters)
    # 这些参数如果丢失，会导致双流无法融合，直接导致色偏和无效输出
    print("\n--- Fusion Parameters Check (Specific) ---")
    fusion_targets = [
        'encoders.0.0.a_l',
        'encoders.0.0.a_r',
        'middle_blks.0.a_l',
        'intro.block_intro.0.seq.0.weight'  # 检查第一层卷积
    ]

    for target in fusion_targets:
        found_key = None
        for k in ckpt_keys:
            if k.endswith(target):
                found_key = k
                break

        status = "✅ Found" if found_key else "❌ MISSING"
        print(f"{target:<35} : {status} (Ckpt key: {found_key})")


if __name__ == "__main__":
    check_keys()