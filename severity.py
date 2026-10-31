import cv2
import numpy as np


def estimate_visual_severity(pil_image):
    """Return a rough color proxy; this is not validated lesion segmentation."""
    rgb = np.asarray(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    leaf = (h >= 20) & (h <= 100) & (s >= 30) & (v >= 30)
    # Candidate yellow/brown pixels inside a rough green-dominant leaf region.
    candidates = ((h >= 20) & (h <= 35) & (s >= 35) & (v >= 25)) | ((s < 45) & (v < 120))
    count = int(np.count_nonzero(leaf))
    if count < 100:
        return {"value": None, "status": "unreliable", "message": "Not enough leaf-colored pixels were detected for this rough estimate."}
    percent = 100.0 * int(np.count_nonzero(leaf & candidates)) / count
    return {
        "value": round(float(np.clip(percent, 0, 100)), 2),
        "status": "experimental_proxy",
        "message": "Experimental color proxy only. Lighting, crop, leaf age, and background can distort it; never use it to calculate pesticide dose.",
    }
