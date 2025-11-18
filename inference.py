 
import argparse, json, os, sys
import numpy as np
import torch
from model import CropHSINet

def eprint(*args):
    print(*args, file=sys.stderr)

def load_model(ckpt_path, device='cpu'):
    if not os.path.exists(ckpt_path):
        eprint(f"[ERR] Model file not found: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    bands = ckpt['bands']; n_classes = ckpt['n_classes']; patch = ckpt.get('patch', 11)
    model = CropHSINet(bands=bands, num_classes=n_classes, heads=4, c2d=64)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    return model, {'bands': bands, 'n_classes': n_classes, 'patch': patch}

def predict_patch(model, patch_np, device='cpu'):
    # Expect (11,11,B)
    if patch_np.ndim != 3:
        eprint(f"[ERR] patch ndim={patch_np.ndim}, expected 3 (11,11,B)")
        sys.exit(1)
    H, W, B = patch_np.shape
    if H != 11 or W != 11:
        eprint(f"[ERR] patch spatial shape={patch_np.shape[:2]}, expected (11,11)")
        sys.exit(1)
    x = torch.from_numpy(np.transpose(patch_np, (2,0,1))[None, None]).float().to(device)
    with torch.no_grad():
        _, _, logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(prob).item())
        conf = float(prob[pred].item())
    return pred, conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--patch', required=False)
    args = ap.parse_args()

    device = 'cpu'
    model, meta = load_model(args.model, device)
    print("[INFO] Loaded model meta:", meta)

    if args.patch:
        if not os.path.exists(args.patch):
            eprint(f"[ERR] Patch file not found: {args.patch}")
            sys.exit(1)
        patch = np.load(args.patch)
        print("[INFO] Loaded patch:", args.patch, "shape:", patch.shape, "dtype:", patch.dtype)
    else:
        patch = np.random.rand(11,11,meta['bands']).astype(np.float32)
        print("[WARN] No --patch provided. Using random patch for smoke test.")

    pred, conf = predict_patch(model, patch, device)
    print(json.dumps({'pred': pred, 'confidence': round(conf, 4)}, indent=2))

if __name__ == "__main__":
    main()
