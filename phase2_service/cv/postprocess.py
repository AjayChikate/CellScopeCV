"""
Post-processing, instance separation, and morphological quantification for nuclei masks.

Pipeline:
1. Probability thresholding (default 0.5)
2. Marker-controlled Watershed segmentation to separate touching/overlapping nuclei:
   - Euclidean distance transform (cv2.distanceTransform)
   - Foreground seed markers from distance peaks (cv2.connectedComponents)
   - Sure background via morphological dilation
   - Watershed algorithm (cv2.watershed)
3. Connected components labeling fallback (cv2.connectedComponentsWithStats)
4. Per-nucleus feature extraction via contours:
   - Area: pixel count and physical area in um^2 (assumed 0.25 um/pixel)
   - Perimeter: arc length
   - Circularity: 4 * pi * area / perimeter^2
   - Eccentricity: sqrt(1 - (minor/major)^2) from fitted ellipse
   - Centroid: (cx, cy) from moments
5. Slide-level aggregates:
   - Count, density (per mm^2 or per pixel)
   - Mean & standard deviation of area, circularity, eccentricity
   - Spatial clustering: nearest-neighbor distance distribution via scipy.spatial.cKDTree
6. Colorized instance overlay rendering
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import cKDTree




class NucleiPostProcessor: #Performs instance separation and quantitative morphological analysis on raw model probability maps.


    def __init__(
        self,
        prob_threshold: float = 0.5,
        min_distance_factor: float = 0.35,
        min_nucleus_area: int = 15,
        pixel_scale_um: float = 0.25,
        method: str = "watershed"
    ):
        self.prob_threshold = prob_threshold
        self.min_distance_factor = min_distance_factor
        self.min_nucleus_area = min_nucleus_area
        self.pixel_scale_um = pixel_scale_um
        self.method = method

    def separate_instances(
        self,
        prob_map: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
       
        h, w = prob_map.shape[:2]


        # Step 1: Binarize probability map
        binary = (prob_map >= self.prob_threshold).astype(np.uint8) * 255
        # Morphological opening to remove isolated 1-2 pixel noise
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3)
        if np.sum(binary) == 0:
            return np.zeros((h, w), dtype=np.uint16)
        if self.method == "connected_components":
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            instance_mask = np.zeros((h, w), dtype=np.uint16)
            current_id = 1
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] >= self.min_nucleus_area:
                    instance_mask[labels == label_idx] = current_id
                    current_id += 1
            return instance_mask


        # Step 2: Marker-Controlled Watershed Algorithm
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist <= 0:
            return np.zeros((h, w), dtype=np.uint16)
        # Foreground markers: detect regional peaks via morphological dilation
        kernel_peak = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated_dist = cv2.dilate(dist_transform, kernel_peak)
        peak_mask = (dist_transform == dilated_dist) & (dist_transform >= self.min_distance_factor * max_dist) & (dist_transform > 2.0)
        sure_fg = cv2.dilate(peak_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        num_markers, markers = cv2.connectedComponents(sure_fg)

        if num_markers <= 1:
            thresh_val = self.min_distance_factor * max_dist
            _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            num_markers, markers = cv2.connectedComponents(sure_fg)

        sure_bg = cv2.dilate(binary, kernel_3, iterations=3)
        unknown = cv2.subtract(sure_bg, (sure_fg > 0).astype(np.uint8) * 255)

        markers = markers + 1
        markers[unknown == 255] = 0

        if rgb_image is not None and rgb_image.ndim == 3:
            ws_input = rgb_image.copy()
        else:
            dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ws_input = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(ws_input, markers)

        instance_mask = np.zeros((h, w), dtype=np.uint16)
        instance_mask[markers > 1] = (markers[markers > 1] - 1).astype(np.uint16)
        instance_mask[binary == 0] = 0

        unassigned = (binary > 0) & (instance_mask == 0)
        if np.any(unassigned):
            dilated = cv2.dilate(instance_mask, kernel_3)
            instance_mask[unassigned] = dilated[unassigned]

        compact_mask = np.zeros((h, w), dtype=np.uint16)
        unique_instances = np.unique(instance_mask)
        current_id = 1

        for inst in unique_instances:
            if inst == 0:
                continue
            inst_region = (instance_mask == inst)
            area_px = int(np.sum(inst_region))
            if area_px >= self.min_nucleus_area:
                compact_mask[inst_region] = current_id
                current_id += 1

        return compact_mask



    def extract_features(
        self,
        instance_mask: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        h, w = instance_mask.shape[:2]
        total_pixels = h * w
        pixel_area_um2 = self.pixel_scale_um ** 2

        unique_ids = np.unique(instance_mask)
        unique_ids = unique_ids[unique_ids > 0]

        nuclei_list: List[Dict[str, Any]] = []
        centroids: List[Tuple[float, float]] = []

        for n_id in unique_ids:
            single_mask = (instance_mask == n_id).astype(np.uint8)
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            area_pixels = float(np.sum(single_mask))
            area_um2 = round(area_pixels * pixel_area_um2, 3)

            perimeter_pixels = float(cv2.arcLength(cnt, True))
            perimeter_um = round(perimeter_pixels * self.pixel_scale_um, 3)

            if perimeter_pixels > 0:
                circularity = (4.0 * math.pi * area_pixels) / (perimeter_pixels ** 2)
                circularity = max(0.0, min(1.0, circularity))
            else:
                circularity = 0.0

            if len(cnt) >= 5:
                try:
                    (cx_e, cy_e), (axis1, axis2), angle = cv2.fitEllipse(cnt)
                    major = max(axis1, axis2)
                    minor = min(axis1, axis2)
                    eccentricity = math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major > 0 else 0.0
                except Exception:
                    eccentricity = 0.0
            else:
                eccentricity = 0.0

            moments = cv2.moments(single_mask)
            if moments['m00'] > 0:
                cx = float(moments['m10'] / moments['m00'])
                cy = float(moments['m01'] / moments['m00'])
            else:
                cx, cy = float(cnt[0][0][0]), float(cnt[0][0][1])

            centroids.append((cx, cy))
            nuclei_list.append({
                "id": int(n_id),
                "area": round(area_pixels, 1),
                "area_um2": area_um2,
                "perimeter": round(perimeter_pixels, 1),
                "perimeter_um": perimeter_um,
                "circularity": round(circularity, 4),
                "eccentricity": round(eccentricity, 4),
                "centroid": [round(cx, 2), round(cy, 2)],
                "centroid_x": round(cx, 2),
                "centroid_y": round(cy, 2)
            })

        df = pd.DataFrame(nuclei_list)

        nucleus_count = len(nuclei_list)
        density_per_px = nucleus_count / total_pixels if total_pixels > 0 else 0.0
        total_area_mm2 = (total_pixels * pixel_area_um2) / 1e6
        density_per_mm2 = nucleus_count / total_area_mm2 if total_area_mm2 > 0 else 0.0

        summary: Dict[str, Any] = {
            "nucleus_count": nucleus_count,
            "density": round(density_per_px, 6),
            "density_per_mm2": round(density_per_mm2, 2),
            "mean_area": round(float(df["area"].mean()), 2) if not df.empty else 0.0,
            "std_area": round(float(df["area"].std()), 2) if len(df) > 1 else 0.0,
            "mean_circularity": round(float(df["circularity"].mean()), 4) if not df.empty else 0.0,
            "std_circularity": round(float(df["circularity"].std()), 4) if len(df) > 1 else 0.0,
            "mean_eccentricity": round(float(df["eccentricity"].mean()), 4) if not df.empty else 0.0,
            "std_eccentricity": round(float(df["eccentricity"].std()), 4) if len(df) > 1 else 0.0,
            "pixel_scale_assumed_um": self.pixel_scale_um
        }

        if nucleus_count >= 2:
            tree = cKDTree(centroids)
            distances, _ = tree.query(centroids, k=2)
            nn_distances = distances[:, 1] * self.pixel_scale_um
            summary["mean_nearest_neighbor_distance_um"] = round(float(np.mean(nn_distances)), 2)
            summary["std_nearest_neighbor_distance_um"] = round(float(np.std(nn_distances)), 2)
        else:
            summary["mean_nearest_neighbor_distance_um"] = None
            summary["std_nearest_neighbor_distance_um"] = None

        return df, summary

    def render_overlay(
        self,
        rgb_image: np.ndarray,
        instance_mask: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        h, w = instance_mask.shape[:2]
        if rgb_image.shape[:2] != (h, w):
            rgb_image = cv2.resize(rgb_image, (w, h))

        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        contour_layer = np.zeros((h, w, 3), dtype=np.uint8)

        unique_ids = np.unique(instance_mask)
        unique_ids = unique_ids[unique_ids > 0]

        np.random.seed(1337)
        palette = np.random.randint(50, 255, size=(len(unique_ids) + 1, 3), dtype=np.uint8)

        for idx, n_id in enumerate(unique_ids):
            inst = (instance_mask == n_id).astype(np.uint8)
            color = palette[idx].tolist()
            color_mask[inst > 0] = color

            contours, _ = cv2.findContours(inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_layer, contours, -1, (255, 255, 255), 1)

            m = cv2.moments(inst)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.circle(contour_layer, (cx, cy), 2, (0, 255, 255), -1)

        foreground = (color_mask > 0).any(axis=-1)
        blended = rgb_image.copy()
        blended[foreground] = cv2.addWeighted(rgb_image[foreground], 1.0 - alpha, color_mask[foreground], alpha, 0)

        has_contour = (contour_layer > 0).any(axis=-1)
        blended[has_contour] = contour_layer[has_contour]

        return blended

