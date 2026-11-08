"""Environment check + a small speed test of the SurgicalSlicing model.

    python code_new/setup/check_env.py [--require-cuda] [--vgg]
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-cuda', action='store_true', help='exit with an error if CUDA is not usable')
    ap.add_argument('--vgg', action='store_true', help='also download the VGG16 ImageNet weights (model.init: vgg16)')
    args = ap.parse_args()

    print(f"python      {platform.python_version()} ({sys.executable})")
    print(f"platform    {platform.platform()}")
    try:
        import torch
        import torchvision
    except ImportError as e:
        sys.exit(f"torch is not installed: {e}")
    import numpy, PIL, yaml   # noqa: F401
    print(f"torch       {torch.__version__}   torchvision {torchvision.__version__}   numpy {numpy.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda        {'yes' if cuda else 'no'}" + (f"  (built for CUDA {torch.version.cuda}, cuDNN "
                                                     f"{torch.backends.cudnn.version()})" if cuda else ''))
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu {i}     {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
    elif args.require_cuda:
        sys.exit("CUDA is not available to torch. Check `nvidia-smi`; if it works, the installed torch build "
                 "does not match the driver - rerun setup_ubuntu_cuda.sh.")
    print(f"cpu threads {torch.get_num_threads()}")

    from surgical.model import GlaucomaNet, count_params
    dev = torch.device('cuda' if cuda else 'cpu')
    model = GlaucomaNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    bs = 32 if cuda else 8
    x = torch.randn(bs, 3, 128, 128, device=dev)
    y = torch.rand(bs, 3, device=dev)
    crit = torch.nn.BCEWithLogitsLoss()
    use_amp = cuda
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

    def step():
        opt.zero_grad()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        loss = crit(out.float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    step()
    if cuda:
        torch.cuda.synchronize()
    n = 10 if cuda else 3
    t0 = time.time()
    for _ in range(n):
        step()
    if cuda:
        torch.cuda.synchronize()
    per_img = (time.time() - t0) / (n * bs)
    print(f"speed       full-size model ({count_params(model):,} params): {per_img * 1000:.1f} ms per training "
          f"image -> ~{per_img * 1000 / 60:.1f} min per 1000 images trained")

    if args.vgg:
        from torchvision.models import vgg16, VGG16_Weights
        vgg16(weights=VGG16_Weights.DEFAULT)
        print("vgg16       ImageNet weights downloaded / cached")
    print("OK")


if __name__ == '__main__':
    main()
