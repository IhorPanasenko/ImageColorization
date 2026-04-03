"""Quick integrity check for torch installation."""
import sys, os, subprocess

base = os.path.join(os.path.dirname(__file__), 'venv/lib/python3.14/site-packages/torch')

# 1. Check all .so/.dylib files are readable
result = subprocess.run(['find', base, '-name', '*.dylib', '-o', '-name', '*.so'],
                        capture_output=True, text=True)
files = [f for f in result.stdout.strip().split('\n') if f]
print(f'Found {len(files)} shared libraries')
bad = []
for f in files:
    try:
        with open(f, 'rb') as fh:
            while fh.read(1024*1024):
                pass
    except Exception as e:
        bad.append((f, str(e)))
        print(f'  CORRUPT: {os.path.basename(f)} -- {e}')
if not bad:
    print('All shared libraries readable from disk.')

# 2. Test torch import
print('\nTesting torch import...')
try:
    import torch
    print(f'torch {torch.__version__} imported OK on {sys.version}')
except Exception as e:
    print(f'torch import FAILED: {e}')
    sys.exit(1)

# 3. Test basic CPU tensor ops
print('\nTesting CPU tensor ops...')
try:
    t = torch.randn(1, 1, 256, 256)
    r = torch.nn.functional.conv2d(t, torch.randn(64, 1, 7, 7), padding=3)
    print(f'Conv2d OK: {t.shape} -> {r.shape}')
except Exception as e:
    print(f'Conv2d FAILED: {e}')

# 4. Test model load with weights_only=False
print('\nTesting torch.load...')
ckpt_dir = os.path.join(os.path.dirname(__file__), 'outputs/checkpoints')
test_ckpt = os.path.join(ckpt_dir, 'baseline_cnn_best.pth')
if os.path.exists(test_ckpt):
    try:
        state = torch.load(test_ckpt, map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            keys = list(state.keys())[:5]
            print(f'Loaded OK, keys: {keys}...')
        else:
            print(f'Loaded OK, type: {type(state)}')
    except Exception as e:
        print(f'torch.load FAILED: {e}')
else:
    print(f'Checkpoint not found: {test_ckpt}')

# 5. Test torchvision import (needed for GlobalHintNet/ResNet18)
print('\nTesting torchvision...')
try:
    from torchvision import models
    print(f'torchvision.models imported OK')
    # Test ResNet18 instantiation (used by GlobalHintNet)
    r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print(f'ResNet18 loaded with pretrained weights OK')
except Exception as e:
    print(f'torchvision FAILED: {e}')

# 6. Test lpips
print('\nTesting lpips...')
try:
    import lpips
    model = lpips.LPIPS(net='alex')
    model.eval()
    a = torch.randn(1, 3, 64, 64)
    b = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        d = model(a, b)
    print(f'LPIPS OK: distance={d.item():.4f}')
except Exception as e:
    print(f'LPIPS FAILED: {e}')

# 7. Test full model build + inference
print('\nTesting full model inference...')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
try:
    from src.models.baseline_cnn import BaselineCNN
    m = BaselineCNN().eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 256, 256))
    print(f'BaselineCNN inference OK: output {out.shape}')
except Exception as e:
    print(f'BaselineCNN FAILED: {e}')

try:
    from src.models.u_net import UNet
    m = UNet().eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 256, 256))
    print(f'UNet inference OK: output {out.shape}')
except Exception as e:
    print(f'UNet FAILED: {e}')

try:
    from src.models.unet_fusion import UNetFusion
    from src.models.global_hints import GlobalHintNet
    g = UNetFusion().eval()
    h = GlobalHintNet(freeze=False).eval()
    L = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        hint = h(L)
        out = g(L, hint)
    print(f'Fusion inference OK: output {out.shape}')
except Exception as e:
    print(f'Fusion FAILED: {e}')

print('\n=== All checks passed ===')
