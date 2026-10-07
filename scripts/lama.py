"""Minimal LaMa (big-lama TorchScript) inpainting wrapper, CPU only.

The model rebuilds the masked area from its surroundings and is far better
than classical fills on textured surfaces such as chain links. Usage:

    from lama import Lama
    fill = Lama("/path/to/big-lama.pt")
    bgr_out = fill(bgr, mask)      # mask: uint8, non-zero = pixels to rebuild
"""
import cv2
import numpy as np
import torch

torch.set_num_threads(max(1, torch.get_num_threads()))


class Lama:
    def __init__(self, weights: str, context: int = 96, min_side: int = 256,
                 full_image_max_side: int = 1024, grow: int = 4):
        """context/min_side: crop around the mask for large frames.
        full_image_max_side: frames up to this size are given to the model
        whole, which lets it continue the product's structure correctly.
        grow: dilate the mask by this many pixels so stamp edge pixels, whose
        colour is a blend of stamp and product, are rebuilt too."""
        self.model = torch.jit.load(weights, map_location="cpu").eval()
        self.context = context
        self.min_side = min_side
        self.full_image_max_side = full_image_max_side
        self.grow = grow

    def _run(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = (mask > 0).astype(np.float32)
        H, W = m.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
        it = torch.from_numpy(img).permute(2, 0, 1)[None]
        mt = torch.from_numpy(m)[None, None]
        with torch.no_grad():
            out = self.model(it, mt)
        out = out[0].permute(1, 2, 0).numpy()[:H, :W]
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def __call__(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.grow:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.grow + 1, 2 * self.grow + 1))
            mask = cv2.dilate(mask, k)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return bgr
        H, W = mask.shape
        if max(H, W) <= self.full_image_max_side:
            res = self._run(bgr, mask)
            out = bgr.copy()
            out[mask > 0] = res[mask > 0]
            return out
        c = self.context
        y0, y1 = max(0, ys.min() - c), min(H, ys.max() + c + 1)
        x0, x1 = max(0, xs.min() - c), min(W, xs.max() + c + 1)
        # give the model enough context: grow the crop to at least min_side
        while (y1 - y0) < self.min_side and (y0 > 0 or y1 < H):
            y0, y1 = max(0, y0 - 8), min(H, y1 + 8)
        while (x1 - x0) < self.min_side and (x0 > 0 or x1 < W):
            x0, x1 = max(0, x0 - 8), min(W, x1 + 8)
        crop, sub = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        res = self._run(crop, sub)
        out = bgr.copy()
        o = out[y0:y1, x0:x1]
        o[sub > 0] = res[sub > 0]
        return out
