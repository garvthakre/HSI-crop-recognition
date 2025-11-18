 
import argparse, os
import numpy as np
import scipy.io as sio
import torch
from model import CropHSINet

def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CropHSINet(bands=ckpt['bands'], num_classes=ckpt['n_classes'], heads=4, c2d=64)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    meta = {'bands': ckpt['bands'], 'n_classes': ckpt['n_classes'], 'patch': ckpt.get('patch', 11)}
    return model, meta

def load_cube_from_mat(mat_path):
    """Load hyperspectral cube from .mat file and normalize per band to [0,1]"""
    m = sio.loadmat(mat_path)
    # Pick the largest ndarray as the cube
    key, arr = max([(k, v) for k, v in m.items() if hasattr(v, 'ndim') and hasattr(v, 'size')], 
                   key=lambda kv: kv[1].size)
    cube = arr
    
    # Handle orientation: some .mat store as (B,H,W), transpose to (H,W,B)
    if cube.ndim == 3 and cube.shape[0] in (200, 270, 300) and cube.shape[2] < 20:
        cube = np.transpose(cube, (1, 2, 0))  # (H,W,B)
    
    cube = cube.astype(np.float32, copy=False)
    H, W, B = cube.shape
    print(f"Loaded cube: {mat_path} | shape: (H={H}, W={W}, B={B})")
    
    # Per-band min-max normalization
    bmin = cube.reshape(-1, B).min(axis=0)
    bmax = cube.reshape(-1, B).max(axis=0)
    brng = np.maximum(bmax - bmin, 1e-8)
    cube = (cube - bmin) / brng
    print(f"Normalized to [0,1]: min={cube.min():.4f}, max={cube.max():.4f}")
    return cube

def sliding_window_predict(cube, model, patch=11, batch=2048, device='cpu'):
    """
    Run sliding window inference over the hyperspectral cube.
    Returns a prediction map with shape (H, W) where each pixel is a class label.
    """
    H, W, B = cube.shape
    h = patch // 2
    pred_map = -np.ones((H, W), dtype=np.int32)  # -1 = no prediction (border)
    
    # Generate centers (skip border pixels)
    centers = [(y, x) for y in range(h, H-h) for x in range(h, W-h)]
    N = len(centers)
    print(f"Processing {N} patches in batches of {batch}...")
    
    device = torch.device(device)
    
    def make_batch(start, end):
        patches = []
        for i in range(start, end):
            y, x = centers[i]
            p = cube[y-h:y+h+1, x-h:x+h+1, :]              # (11,11,B)
            patches.append(np.transpose(p, (2, 0, 1)))     # (B,11,11)
        arr = np.stack(patches, 0)[:, None]                # (n,1,B,11,11)
        return torch.from_numpy(arr).float().to(device), start, end
    
    with torch.no_grad():
        for i in range(0, N, batch):
            X, s, e = make_batch(i, min(i+batch, N))
            _, _, logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            k = 0
            for j in range(s, e):
                y, x = centers[j]
                pred_map[y, x] = preds[k]
                k += 1
            
            if (i // batch) % 10 == 0:
                print(f"  Processed {min(i+batch, N)}/{N} patches...")
    
    return pred_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to saved .pth model')
    ap.add_argument('--mat', required=True, help='Path to WHU .mat scene (e.g., WHU_Hi_LongKou.mat)')
    ap.add_argument('--out', default='pred_map.npy', help='Output .npy for prediction map')
    ap.add_argument('--batch', type=int, default=2048, help='Batch size for inference')
    args = ap.parse_args()
    
    device = 'cpu'
    print("Loading model...")
    model, meta = load_model(args.model, device)
    print(f"Model meta: {meta}")
    
    print("\nLoading scene...")
    cube = load_cube_from_mat(args.mat)
    
    if cube.shape[2] != meta['bands']:
        print(f"ERROR: Scene bands ({cube.shape[2]}) != model bands ({meta['bands']})")
        return
    
    print("\nRunning inference...")
    pred_map = sliding_window_predict(cube, model, patch=meta['patch'], batch=args.batch, device=device)
    
    np.save(args.out, pred_map)
    print(f"\n✓ Saved: {args.out} | shape: {pred_map.shape}")
    print(f"Predicted classes: {sorted(set(pred_map[pred_map >= 0]))}")

if __name__ == "__main__":
    main()
