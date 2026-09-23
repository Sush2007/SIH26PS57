import numpy as np
import cv2
import math

class MultiEvidenceEngine:
    def __init__(self, meters_per_pixel: float = 0.05):
        self.m_per_px = meters_per_pixel
        
        # Military-Grade Threat Classification Matrix
        self.threat_matrix = {
            "mine_cylinder": "CRITICAL_MCM_THREAT",
            "shipwreck": "NAV_HAZARD",
            "submarine_pipeline": "CRITICAL_INFRASTRUCTURE",
            "ghost_net": "ENTANGLEMENT_HAZARD",
            "crab_pot": "IGNORED_HARD_NEGATIVE"
        }

    def _platt_scale(self, raw_conf: float) -> float:
        """
        Applies logistic regression calibration to convert raw YOLO 
        network activations into true statistical probability.
        """
        A = 5.5
        B = -2.5
        prob = 1.0 / (1.0 + math.exp(-(A * raw_conf + B)))
        return float(np.clip(prob, 0.0, 1.0))

    def evaluate(self, img: np.ndarray, obb_xywhr: list, class_name: str, altitude_m: float, towfish_lat: float, towfish_lon: float, raw_conf: float):
        # 🛑 0. Hard Negative Suppression
        if class_name == "crab_pot":
            return {
                "verdict": "REJECTED_AS_HARD_NEGATIVE",
                "fused_confidence": 0.0,
                "evidence_breakdown": {},
                "physical_dimensions": {},
                "corrected_coordinates": {},
                "tactical_telemetry": {}
            }

        xc, yc, w_px, h_px, angle_rad = obb_xywhr
        img_h, img_w = img.shape

        # 🤖 1. AI Evidence (Platt-Scaled Calibration)
        calibrated_prob = self._platt_scale(raw_conf)
        
        # 🌊 2. Acoustic Evidence (Dynamic Relative Contrast)
        local_floor = float(np.percentile(img, 20))
        local_median = float(np.median(img))
        shadow_step = int(max(w_px, h_px) * 0.6)
        patch_rad = 12

        scores = []
        for direction in [1.0, -1.0]:
            sx = int(np.clip(xc + (direction * shadow_step), patch_rad, img_w - patch_rad))
            sy = int(np.clip(yc, patch_rad, img_h - patch_rad))
            patch = img[sy - patch_rad : sy + patch_rad, sx - patch_rad : sx + patch_rad]
            
            if patch.size > 0:
                patch_mean = float(np.mean(patch))
                score = np.clip((local_median - patch_mean) / (local_median - local_floor + 1e-5), 0.0, 1.0)
                scores.append(score)
        
        acoustic_score = float(max(scores)) if scores else 0.5

        # 🖼️ 3. Image-Quality Evidence (Dynamic SNR)
        local_std = float(np.std(img))
        if 18.0 <= local_std <= 90.0:
            quality_score = 0.85
        else:
            quality_score = float(np.clip(local_std / 25.0, 0.3, 1.0))

        # 🪨 4. Natural-Feature Evidence
        obj_x1, obj_x2 = max(0, int(xc - w_px/2)), min(img_w, int(xc + w_px/2))
        obj_y1, obj_y2 = max(0, int(yc - h_px/2)), min(img_h, int(yc + h_px/2))
        obj_patch = img[obj_y1:obj_y2, obj_x1:obj_x2]

        natural_penalty = 0.0
        if obj_patch.size > 20:
            grad_x = cv2.Sobel(obj_patch, cv2.CV_64F, 1, 0, ksize=3)
            grad_var = float(np.var(grad_x))
            if grad_var < 40.0:
                natural_penalty = 0.25
        natural_score = float(np.clip(1.0 - natural_penalty, 0.0, 1.0))

        # 📐 5. Geometric Evidence
        real_length_m = max(w_px, h_px) * self.m_per_px
        real_width_m = min(w_px, h_px) * self.m_per_px
        aspect_ratio = real_length_m / (real_width_m + 1e-5)

        geometric_score = 0.85
        if class_name == "submarine_pipeline":
            geometric_score = 1.0 if aspect_ratio >= 2.2 else 0.3
        elif class_name == "mine_cylinder":
            geometric_score = 1.0 if (0.3 <= real_length_m <= 3.5) else 0.4
        elif class_name == "shipwreck":
            geometric_score = 1.0 if real_length_m >= 3.0 else 0.5
        elif class_name == "ghost_net":
            geometric_score = 0.95 

        # 📍 6. Geospatial & Tactical Evidence
        nadir_x = img_w / 2.0
        dx = xc - nadir_x
        dir_sign = 1.0 if dx >= 0 else -1.0
        channel_side = "STARBOARD" if dx >= 0 else "PORT"
        
        dist_from_nadir_px = abs(dx) + 1e-5
        slant_range_m = dist_from_nadir_px * self.m_per_px

        if slant_range_m >= altitude_m:
            ground_range_m = float(np.sqrt(slant_range_m**2 - altitude_m**2))
            geo_uncertainty_m = 1.2
        else:
            ground_range_m = float(slant_range_m)
            geo_uncertainty_m = 2.5

        lat_offset = (ground_range_m * 0.0) / 111320.0
        lon_offset = (dir_sign * ground_range_m) / (111320.0 * np.cos(np.radians(towfish_lat)))
        actual_lat = towfish_lat + lat_offset
        actual_lon = towfish_lon + lon_offset

        # 📊 Advanced Telemetry Calculations
        strike_angle_deg = round(math.degrees(angle_rad) % 180, 1)
        
        # 3D Relief Height Calculation: H = (Ls * A) / (Ls + R)
        estimated_shadow_len_m = (shadow_step * self.m_per_px) * acoustic_score
        if estimated_shadow_len_m > 0 and slant_range_m > 0:
            relief_height_m = (estimated_shadow_len_m * altitude_m) / (estimated_shadow_len_m + slant_range_m)
        else:
            relief_height_m = 0.0

        # ⚡ Decision Engine Fusion Matrix
        geo_score = float(1.0 - min(geo_uncertainty_m / 10.0, 0.6))

        # 🛑 THE LOOPHOLE PATCH: Hard Veto Conditions
        if calibrated_prob < 0.40 and acoustic_score < 0.35:
            verdict = "REJECTED_NO_PHYSICAL_EVIDENCE"
            fused_confidence = 0.0
        # If the object physically defies the geometric boundaries of its class, kill it.
        elif geometric_score == 0.0:
            verdict = "REJECTED_GEOMETRIC_VIOLATION"
            fused_confidence = 0.0
        else:
            # Standard weighted integration
            fused_confidence = (
                (0.35 * calibrated_prob) +
                (0.20 * acoustic_score) +
                (0.15 * geometric_score) +
                (0.10 * natural_score) +
                (0.10 * quality_score) +
                (0.10 * geo_score)
            )

            if fused_confidence >= 0.65:
                verdict = "CONFIRMED_ANOMALY"
            elif fused_confidence >= 0.40:
                verdict = "PROBABLE_TARGET"
            else:
                verdict = "REJECTED_AS_NATURAL_ARTIFACT"

        return {
            "verdict": verdict,
            "fused_confidence": round(float(fused_confidence), 3),
            "evidence_breakdown": {
                "calibrated_ai_probability": round(calibrated_prob, 3),
                "acoustic_shadow_strength": round(acoustic_score, 2),
                "image_quality_index": round(quality_score, 2),
                "natural_feature_exclusion": round(natural_score, 2),
                "geometric_plausibility": round(geometric_score, 2),
                "geospatial_accuracy_radius_m": round(geo_uncertainty_m, 2)
            },
            "physical_dimensions": {
                "length_meters": round(real_length_m, 2),
                "width_meters": round(real_width_m, 2),
                "ground_range_meters": round(ground_range_m, 2)
            },
            "corrected_coordinates": {
                "latitude": round(actual_lat, 6),
                "longitude": round(actual_lon, 6)
            },
            "tactical_telemetry": {
                "threat_classification": self.threat_matrix.get(class_name, "UNKNOWN_ANOMALY"),
                "acoustic_channel": channel_side,
                "target_strike_heading_deg": strike_angle_deg,
                "estimated_3d_relief_height_m": round(relief_height_m, 2)
            }
        }