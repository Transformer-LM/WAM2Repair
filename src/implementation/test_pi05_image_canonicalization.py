"""CPU-only check for the exact 2x raw-WAM downsampling rule used by the probe."""
import numpy as np
import torch
import torch.nn.functional as F


def canonical(x):
    return np.rint(x.astype(np.float32).reshape(112, 2, 224, 2, 3).mean(axis=(1, 3))).clip(0, 255).astype(np.uint8)


rng = np.random.default_rng(20260918)
x = rng.integers(0, 256, size=(224, 448, 3), dtype=np.uint8)
reference = F.interpolate(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float(), size=(112, 224), mode="bilinear", align_corners=False)
reference = torch.round(reference).clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).numpy()
np.testing.assert_array_equal(canonical(x), reference)
print("PASS: exact 2x bilinear canonicalization")
