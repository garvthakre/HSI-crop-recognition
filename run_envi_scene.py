 
import argparse, os
import numpy as np
import spectral as sp
import torch
from model import CropHSINet

def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CropHSINet(bands=ckpt['bands'], num_classes=ckpt['n_classes'], heads=4, c2d=64)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    meta = {'bands': ckpt['bands'], 'n_classes': ckpt['n_classes'], 'patch': ckpt.get('patch', 11)}
    return model, meta

def load_envi_cube(hdr_path):
    img = sp.envi.open(hdr_path)              # finds raw via header
    cube = img.load().astype(np.float32)      # (H,W,B)
    H, W, B = cube.shape
    # per-band min-max normalization
    bmin = cube.reshape(-1, B).min(axis=0)
    bmax = cube.reshape(-1, B).max(axis=0)
    brng = np.maximum(bmax - bmin, 1e-8)
    cube = (cube - bmin) / brng
    return cube

def sliding_window_predict(cube, model, patch=11, batch=2048, device='cpu'):
    H, W, B = cube.shape
    h = patch // 2
    pred_map = -np.ones((H, W), dtype=np.int32)
    centers = [(y, x) for y in range(h, H-h) for x in range(h, W-h)]
    device = torch.device(device)
    def make_batch(s, e):
        patches = []
        for i in range(s, e):
            y, x = centers[i]
            p = cube[y-h:y+h+1, x-h:x+h+1, :]            # (11,11,B)
            patches.append(np.transpose(p, (2,0,1)))     # (B,11,11)
        arr = np.stack(patches, 0)[:, None]              # (n,1,B,11,11)
        return torch.from_numpy(arr).float().to(device), s, e
    with torch.no_grad():
        for i in range(0, len(centers), batch):
            X, s, e = make_batch(i, min(i+batch, len(centers)))
            _, _, logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            k = 0
            for j in range(s, e):
                y, x = centers[j]
                pred_map[y, x] = preds[k]; k += 1
    return pred_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to .pth')
    ap.add_argument('--hdr', required=True, help='Path to ENVI .hdr header')
    ap.add_argument('--out', default='pred_map.npy', help='Output .npy path')
    args = ap.parse_args()

    device = 'cpu'
    model, meta = load_model(args.model, device)
    cube = load_envi_cube(args.hdr)
    assert cube.shape[2] == meta['bands'], f"Scene bands {cube.shape[2]} != model bands {meta['bands']}"
    pred_map = sliding_window_predict(cube, model, patch=meta['patch'], device=device)
    np.save(args.out, pred_map)
    print("Saved:", args.out, "| shape:", pred_map.shape)

if __name__ == "__main__":
    main()
