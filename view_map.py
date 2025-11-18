 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='pred_map.npy', help='Path to prediction map .npy')
    args = ap.parse_args()
    
    pred = np.load(args.map)
    
    # Count pixels per class
    unique, counts = np.unique(pred[pred >= 0], return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt:,} pixels")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(pred, cmap='tab20', interpolation='nearest')
    plt.colorbar(label='Class ID')
    plt.title('Crop Classification Map')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig('pred_map.png', dpi=150)
    print("\n✓ Saved visualization: pred_map.png")
    plt.show()

if __name__ == "__main__":
    main()
