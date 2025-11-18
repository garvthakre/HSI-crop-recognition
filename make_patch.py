 
import numpy as np

B = 270      # bands must match model meta
P = 11       # patch size
patch = np.random.rand(P, P, B).astype(np.float32)  # normalized [0,1]
np.save('sample_patch.npy', patch)
print("Wrote sample_patch.npy with shape", patch.shape)
