"""
Sonar Image Validator — Rejects non-sonar images before they reach the YOLO model.

Uses multiple heuristic checks based on the known statistical properties of
side-scan sonar imagery to distinguish genuine sonar scans from photographs,
documents, screenshots, and other non-sonar inputs.
"""

import cv2
import numpy as np


def validate_sonar_image(img_gray: np.ndarray) -> dict:
    """
    Runs a battery of heuristic tests on a grayscale image to determine
    whether it is likely a side-scan sonar scan.

    Returns:
        dict with keys:
            - is_valid (bool): True if the image passes all checks
            - confidence (float): 0.0 to 1.0 sonar-likeness score
            - reason (str): Human-readable rejection reason (empty if valid)
            - checks (dict): Individual check results for debugging
    """
    checks = {}
    reasons = []

    h, w = img_gray.shape

    # ─── Check 1: Aspect Ratio ───
    # Sonar images are typically wide panoramic strips or square tiles.
    # Portrait photos of documents are tall and narrow.
    aspect = w / h if h > 0 else 0
    # Reject extremely tall portrait images (aspect < 0.4)
    checks["aspect_ratio"] = round(aspect, 2)
    if aspect < 0.35:
        reasons.append("Image aspect ratio is too tall/narrow for a sonar scan")

    # ─── Check 2: Color Channel Detection ───
    # This check is done before grayscale conversion in the caller,
    # but we can detect if the grayscale has too much color variation
    # by checking the histogram entropy. Sonar images have a specific
    # intensity distribution — they are NOT uniformly distributed.

    # ─── Check 3: Histogram Distribution ───
    # Sonar images have a characteristic bimodal histogram:
    # a large dark peak (water column/shadows) and a mid-tone peak (seafloor).
    # Photos of documents have a very bright peak (white paper) + dark peak (text).
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    # Percentage of pixels that are very bright (>220) — documents/photos have lots
    bright_fraction = float(np.sum(hist_norm[220:]))
    checks["bright_pixel_fraction"] = round(bright_fraction, 3)
    if bright_fraction > 0.25:
        reasons.append(
            f"Too many bright pixels ({bright_fraction:.0%}). "
            "Sonar images are predominantly dark — this looks like a photo or document"
        )

    # Percentage of pixels that are very dark (<30) — sonar has a moderate amount
    dark_fraction = float(np.sum(hist_norm[:30]))
    checks["dark_pixel_fraction"] = round(dark_fraction, 3)

    # ─── Check 4: Mean Intensity ───
    # Sonar images are generally dark (mean 40-160).
    # Photos of documents are bright (mean > 180).
    mean_val = float(np.mean(img_gray))
    checks["mean_intensity"] = round(mean_val, 1)
    if mean_val > 200:
        reasons.append(
            f"Mean pixel intensity is {mean_val:.0f}/255 — far too bright for sonar. "
            "Sonar images are predominantly dark"
        )

    # ─── Check 5: Edge Density (Canny) ───
    # Documents and photos of text have extremely high edge density with sharp high-frequency transitions.
    # Sonar backscatter has natural acoustic speckle noise that produces fine textures.
    # We use a higher threshold (100, 200) to isolate true structural edges from speckle.
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (h * w)
    checks["edge_density"] = round(edge_density, 4)
    # Reject only if edge density is abnormally high (e.g. dense printed text or diagrams)
    if edge_density > 0.35:
        reasons.append(
            f"Edge density is {edge_density:.1%} — excessive sharp high-contrast edges. "
            "This looks like a photograph of text or a document, not a sonar scan"
        )

    # ─── Check 6: Vertical Symmetry ───
    # Side-scan sonar often has approximate left-right symmetry around
    # the nadir line (center column). Photos and documents do not.
    left_half = img_gray[:, : w // 2]
    right_half = img_gray[:, w // 2 :]
    # Trim to equal widths
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = np.flip(right_half[:, :min_w], axis=1)
    symmetry = float(
        1.0
        - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
    )
    checks["vertical_symmetry"] = round(symmetry, 3)

    # ─── Check 7: Nadir Line Detection ───
    # Many sonar images have a distinct dark vertical band in the center (nadir).
    # We check if the center column strip is darker than the flanks.
    center_strip = img_gray[:, w // 2 - 5 : w // 2 + 5]
    flank_left = img_gray[:, w // 4 - 10 : w // 4 + 10]
    flank_right = img_gray[:, 3 * w // 4 - 10 : 3 * w // 4 + 10]
    center_mean = float(np.mean(center_strip))
    flank_mean = float((np.mean(flank_left) + np.mean(flank_right)) / 2)
    nadir_contrast = (flank_mean - center_mean) / (flank_mean + 1e-5)
    checks["nadir_contrast"] = round(nadir_contrast, 3)

    # ─── Check 8: Texture Uniformity (Local Std Dev) ───
    # Sonar backscatter has a characteristic speckle texture with moderate local variance.
    # Photos have very high local variance (sharp color transitions).
    # Blank/uniform images have very low variance.
    local_std = cv2.blur(
        cv2.absdiff(img_gray, cv2.blur(img_gray, (15, 15))).astype(np.float32),
        (15, 15),
    )
    mean_local_std = float(np.mean(local_std))
    checks["mean_local_texture"] = round(mean_local_std, 2)
    if mean_local_std > 55:
        reasons.append(
            "Image texture is too sharp and varied for sonar — "
            "sonar images have a soft speckle pattern, not sharp photo details"
        )

    # ─── Scoring ───
    # Each passing check contributes to the confidence score
    score = 1.0
    if bright_fraction > 0.15:
        score -= min(bright_fraction * 1.5, 0.35)
    if mean_val > 170:
        score -= min((mean_val - 170) / 100, 0.25)
    if edge_density > 0.25:
        score -= min((edge_density - 0.25) * 2, 0.30)
    if mean_local_std > 50:
        score -= min((mean_local_std - 50) / 50, 0.25)
    if aspect < 0.4:
        score -= 0.15

    # Bonus for sonar-like properties
    if nadir_contrast > 0.05:
        score += 0.08
    if symmetry > 0.7:
        score += 0.05
    if 40 < mean_val < 140 and dark_fraction > 0.15:
        score += 0.10

    score = float(np.clip(score, 0.0, 1.0))
    checks["sonar_confidence_score"] = round(score, 3)

    is_valid = len(reasons) == 0 and score >= 0.40

    return {
        "is_valid": is_valid,
        "confidence": score,
        "reason": reasons[0] if reasons else "",
        "checks": checks,
    }
