"""
Build the evaluation pairs from the raw LROC frames in datasets/real/raw.

Each pair is a real lunar image with a *synthetic* geometric transform applied
to produce the reference. The transform is written to datasets/manifest.json as
a 3x3 matrix, which gives every pair a known ground truth: registration
accuracy can be measured against the true transform rather than against the
estimator's own residual.

Run this before register.py or main.py if datasets/manifest.json is missing.
"""

import json
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUT_DIR = os.path.join(DATASET_DIR, "real")
RAW_DIR = os.path.join(OUT_DIR, "raw")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")

RAW_PROVENANCE = "NASA / LRO Lunar Reconnaissance Orbiter Camera (LROC), public domain"


def affine_to_homography(M):
    """Promote a 2x3 affine matrix to a 3x3 homography."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def similarity_matrix(width, height, rotation_deg, scale, tx, ty):
    M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation_deg, scale)
    M[:, 2] += [tx, ty]
    return M


def apply_illumination_change(img, gamma, gradient_strength):
    """
    Approximate a sun-angle difference: a global tone change plus a linear
    brightness ramp across the frame. Crude next to a real photometric model,
    but it breaks the intensity constancy that naive matchers rely on.
    """
    normalized = img.astype(np.float32) / 255.0
    adjusted = np.power(normalized, gamma)

    h, w = img.shape[:2]
    ramp = np.linspace(-gradient_strength, gradient_strength, w, dtype=np.float32)
    ramp = np.tile(ramp, (h, 1))
    if adjusted.ndim == 3:
        ramp = ramp[:, :, None]

    return np.clip((adjusted + ramp) * 255.0, 0, 255).astype(np.uint8)


def build_same_scale_pair(raw_name, crop, rotation_deg, scale, tx, ty, illumination=None):
    """Source and reference at the same GSD, related by a known similarity transform."""
    raw_path = os.path.join(RAW_DIR, raw_name)
    if not os.path.exists(raw_path):
        return None

    img = cv2.imread(raw_path)
    y, x, h, w = crop
    src = img[y : y + h, x : x + w].copy()
    if src.shape[0] != h or src.shape[1] != w:
        raise ValueError(f"{raw_name}: crop {crop} exceeds image bounds {img.shape[:2]}")

    M = similarity_matrix(w, h, rotation_deg, scale, tx, ty)
    ref = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    if illumination is not None:
        ref = apply_illumination_change(ref, **illumination)

    return {
        "src": src,
        "ref": ref,
        "ground_truth_H": affine_to_homography(M).tolist(),
        "src_gsd": 1.0,
        "ref_gsd": 1.0,
        "transform": {
            "rotation_deg": rotation_deg,
            "scale": scale,
            "translation_px": [tx, ty],
            "illumination": illumination,
        },
    }


def build_cross_scale_pair(raw_name, crop, rotation_deg, scale, tx, ty, decimation):
    """
    Source at a coarser GSD than the reference, to exercise stage 1 resampling.

    The reference keeps the full resolution of the raw frame; the source is
    decimated by `decimation`, emulating a lower-resolution instrument imaging
    the same ground. Ground truth composes the decimation with the transform.
    """
    raw_path = os.path.join(RAW_DIR, raw_name)
    if not os.path.exists(raw_path):
        return None

    img = cv2.imread(raw_path)
    y, x, h, w = crop
    full = img[y : y + h, x : x + w].copy()
    if full.shape[0] != h or full.shape[1] != w:
        raise ValueError(f"{raw_name}: crop {crop} exceeds image bounds {img.shape[:2]}")

    M = similarity_matrix(w, h, rotation_deg, scale, tx, ty)
    ref = cv2.warpAffine(full, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    src = cv2.resize(
        full, (w // decimation, h // decimation), interpolation=cv2.INTER_AREA
    )

    # A source pixel maps to full-resolution crop coordinates by scaling up,
    # then into the reference frame by M.
    upscale = np.diag([float(decimation), float(decimation), 1.0])
    ground_truth = affine_to_homography(M) @ upscale

    return {
        "src": src,
        "ref": ref,
        "ground_truth_H": ground_truth.tolist(),
        "src_gsd": float(decimation),
        "ref_gsd": 1.0,
        "transform": {
            "rotation_deg": rotation_deg,
            "scale": scale,
            "translation_px": [tx, ty],
            "source_decimation": decimation,
        },
    }


def build_cross_modal_pair(raw_name, crop, rotation_deg, scale, tx, ty, decimation):
    """
    A stand-in for an optical-to-infrared pair.

    Cross-modal matching fails because *intensity constancy* fails: a feature
    bright at one wavelength can be dark at another. The reference here is
    contrast-inverted and given a strong non-linear tone curve, which breaks
    that assumption outright — NCC between the two frames goes to roughly -1.
    It is also decimated to a coarser GSD, since real cross-instrument pairs
    differ in scale as well.

    What this does NOT reproduce: real spectral response, thermal emission, or
    features genuinely present in one band and absent in the other. It exercises
    the failure mode the structural pipeline is built to survive; it is not a
    substitute for real IIRS data.
    """
    raw_path = os.path.join(RAW_DIR, raw_name)
    if not os.path.exists(raw_path):
        return None

    img = cv2.imread(raw_path)
    y, x, h, w = crop
    full = img[y : y + h, x : x + w].copy()
    if full.shape[0] != h or full.shape[1] != w:
        raise ValueError(f"{raw_name}: crop {crop} exceeds image bounds {img.shape[:2]}")

    M = similarity_matrix(w, h, rotation_deg, scale, tx, ty)
    ref = cv2.warpAffine(full, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # Invert, then an S-curve. Inversion alone flips polarity; the curve makes
    # the mapping non-linear so it cannot be undone by a global rescale.
    inverted = 255 - ref
    curve = np.clip(((np.arange(256) / 255.0) ** 0.55) * 255.0, 0, 255).astype(np.uint8)
    ref = cv2.LUT(inverted, curve)

    src = cv2.resize(
        full, (w // decimation, h // decimation), interpolation=cv2.INTER_AREA
    )

    upscale = np.diag([float(decimation), float(decimation), 1.0])
    ground_truth = affine_to_homography(M) @ upscale

    return {
        "src": src,
        "ref": ref,
        "ground_truth_H": ground_truth.tolist(),
        "src_gsd": float(decimation),
        "ref_gsd": 1.0,
        "transform": {
            "rotation_deg": rotation_deg,
            "scale": scale,
            "translation_px": [tx, ty],
            "source_decimation": decimation,
            "radiometry": "contrast inversion + gamma 0.55 tone curve",
        },
    }


PAIR_SPECS = [
    {
        "id": "cleomedes",
        "name": "Cleomedes crater",
        "summary": "Small rotation and translation at matched resolution.",
        "condition": "same_sensor",
        "raw": "cleomedes.jpg",
        "builder": build_same_scale_pair,
        "params": dict(
            crop=(120, 150, 480, 640), rotation_deg=2.2, scale=1.03, tx=14, ty=-10
        ),
    },
    {
        "id": "humboldt",
        "name": "Humboldt crater",
        "summary": "Geometric offset combined with a synthetic illumination change.",
        "condition": "sun_angle_delta",
        "raw": "humboldt.jpg",
        "builder": build_same_scale_pair,
        "params": dict(
            crop=(200, 300, 480, 640),
            rotation_deg=-3.8,
            scale=0.97,
            tx=-18,
            ty=16,
            illumination=dict(gamma=0.65, gradient_strength=0.12),
        ),
    },
    {
        "id": "bell",
        "name": "Bell crater",
        "summary": "Low-contrast regolith with a sub-degree rotation.",
        "condition": "same_sensor",
        "raw": "bell.jpg",
        "builder": build_same_scale_pair,
        "params": dict(
            crop=(40, 25, 360, 400), rotation_deg=1.5, scale=1.01, tx=8, ty=-6
        ),
    },
    {
        "id": "humboldt_crossscale",
        "name": "Humboldt crater (4:1 scale)",
        "summary": "Source decimated 4x against a full-resolution reference.",
        "condition": "cross_sensor",
        "raw": "humboldt.jpg",
        "builder": build_cross_scale_pair,
        "params": dict(
            crop=(400, 500, 960, 1280),
            rotation_deg=1.8,
            scale=1.0,
            tx=12,
            ty=-9,
            decimation=4,
        ),
    },
    {
        "id": "humboldt_crossmodal",
        "name": "Humboldt crater (cross-modal proxy)",
        "summary": "Contrast-inverted reference at 4:1 scale — intensity matching fails here.",
        "condition": "cross_modal",
        "raw": "humboldt.jpg",
        "builder": build_cross_modal_pair,
        "params": dict(
            crop=(400, 500, 960, 1280),
            rotation_deg=2.4,
            scale=1.0,
            tx=-14,
            ty=11,
            decimation=4,
        ),
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []

    for spec in PAIR_SPECS:
        built = spec["builder"](spec["raw"], **spec["params"])
        if built is None:
            print(f"[skip] {spec['id']}: raw frame {spec['raw']} not found")
            continue

        src_name = f"{spec['id']}_source.png"
        ref_name = f"{spec['id']}_reference.png"
        cv2.imwrite(os.path.join(OUT_DIR, src_name), built["src"])
        cv2.imwrite(os.path.join(OUT_DIR, ref_name), built["ref"])

        manifest.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "summary": spec["summary"],
                "condition": spec["condition"],
                "source_path": f"datasets/real/{src_name}",
                "reference_path": f"datasets/real/{ref_name}",
                "source_url": f"/datasets/real/{src_name}",
                "reference_url": f"/datasets/real/{ref_name}",
                "source_size": [built["src"].shape[1], built["src"].shape[0]],
                "reference_size": [built["ref"].shape[1], built["ref"].shape[0]],
                "src_gsd": built["src_gsd"],
                "ref_gsd": built["ref_gsd"],
                "gsd_ratio": f"{built['src_gsd'] / built['ref_gsd']:g}:1",
                "ground_truth_H": built["ground_truth_H"],
                "applied_transform": built["transform"],
                "synthetic": True,
                "provenance": {
                    "raw_frame": f"datasets/real/raw/{spec['raw']}",
                    "credit": RAW_PROVENANCE,
                    "note": (
                        "Reference generated from the source frame by a known "
                        "transform. Imagery is real; the geometric and photometric "
                        "differences are synthetic and exactly known."
                    ),
                },
            }
        )

        sw, sh = built["src"].shape[1], built["src"].shape[0]
        rw, rh = built["ref"].shape[1], built["ref"].shape[0]
        print(f"[ok] {spec['id']:22s} source {sw}x{sh}  reference {rw}x{rh}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n{len(manifest)} pairs written; manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
