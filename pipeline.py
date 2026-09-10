"""
Six-stage image registration pipeline for lunar orbital imagery.

Stage 1  Normalize  CLAHE contrast equalisation + GSD-aware resampling
Stage 2  Match      LoFTR dense matching, with a SIFT/ORB fallback
Stage 3  Verify     MAGSAC++ homography estimation and outlier rejection
Stage 4  Align      Projective warp into the reference coordinate frame
Stage 5  Refine     NCC template matching with sub-pixel peak interpolation
Stage 6  Report     Condition-stratified evaluation against per-condition budgets

Coordinate frames
-----------------
Stage 1 may resample the reference so that both images share a ground sample
distance. Stages 2, 3 and 5 then all work in that *common frame*. This matters
most for stage 5: correlating a source patch against a reference patch covering
a different ground area produces meaningless peaks.

`execute_pipeline` maps the refined homography out of the common frame into
full-resolution reference pixels, so the matrix it reports, the warp it
delivers and every error figure are in the frame the product is consumed in.
"""

import time

import cv2
import numpy as np

_LOFTR_MODEL = None
_LOFTR_LOAD_ATTEMPTED = False


def get_loftr_model():
    """Load the pretrained LoFTR model once and cache it. Returns None if unavailable."""
    global _LOFTR_MODEL, _LOFTR_LOAD_ATTEMPTED

    if _LOFTR_MODEL is not None:
        return _LOFTR_MODEL
    if _LOFTR_LOAD_ATTEMPTED:
        return None

    _LOFTR_LOAD_ATTEMPTED = True
    try:
        import torch
        import kornia.feature as KF

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LOFTR_MODEL = KF.LoFTR(pretrained="outdoor").to(device).eval()
        print(f"[pipeline] LoFTR loaded on {device}")
        return _LOFTR_MODEL
    except Exception as exc:
        print(f"[pipeline] LoFTR unavailable ({exc}); using SIFT/ORB fallback")
        return None


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


# ---------------------------------------------------------------------------
# Stage 1: normalize
# ---------------------------------------------------------------------------
def _apply_gamma(img, gamma):
    """Gamma correction through a 256-entry lookup rather than per-pixel pow."""
    if abs(gamma - 1.0) < 0.01:
        return img
    table = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)
    return cv2.LUT(img, table.astype(np.uint8))


def stage1_normalize(
    src_img,
    ref_img,
    src_gsd=1.0,
    ref_gsd=1.0,
    use_clahe=True,
    gamma=1.0,
    denoise=0.0,
    representation="intensity",
):
    """
    Equalise local contrast and bring both images to a common ground sample
    distance.

    The reference is resampled by ``ref_gsd / src_gsd``. A ratio of 1.0 (the
    default) leaves it untouched, so callers that do not know the sensor GSDs
    get a straight 1:1 comparison rather than a silent rescale.

    Returns the achieved x/y scale factors, which may differ slightly from the
    requested ratio because output dimensions are integers.
    """
    t0 = time.time()

    src_norm = _to_gray(src_img)
    ref_norm = _to_gray(ref_img)

    if denoise > 0:
        # Bilateral rather than Gaussian: it suppresses sensor noise without
        # softening the crater rims the matcher keys on.
        diameter = max(3, int(round(denoise * 4)) | 1)
        src_norm = cv2.bilateralFilter(src_norm, diameter, denoise * 25, denoise * 25)
        ref_norm = cv2.bilateralFilter(ref_norm, diameter, denoise * 25, denoise * 25)

    if gamma != 1.0:
        src_norm = _apply_gamma(src_norm, gamma)
        ref_norm = _apply_gamma(ref_norm, gamma)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        src_norm = clahe.apply(src_norm)
        ref_norm = clahe.apply(ref_norm)

    ratio = ref_gsd / src_gsd if src_gsd > 0 and ref_gsd > 0 else 1.0

    if abs(ratio - 1.0) < 0.01:
        ref_resampled = ref_norm
        scale_x = scale_y = 1.0
    else:
        h, w = ref_norm.shape[:2]
        new_w = max(32, int(round(w * ratio)))
        new_h = max(32, int(round(h * ratio)))
        interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC
        ref_resampled = cv2.resize(ref_norm, (new_w, new_h), interpolation=interp)
        # Derive the achieved scale from the actual output size rather than the
        # requested ratio, so the inverse mapping in stage 3 stays exact.
        scale_x = new_w / w
        scale_y = new_h / h

    # Cross-modal reduction. Replacing both images with a phase-congruency map
    # removes the intensity relationship the matchers assume, leaving only
    # structure — which is what survives a change of wavelength. Applied after
    # resampling so both maps are computed at the same scale.
    if representation == "structural":
        import modality

        # Note whether resampling actually produced a distinct array before
        # rebinding either name, so the shared case stays shared.
        resampled_is_reference = ref_resampled is ref_norm

        src_norm = modality.structural_map(src_norm)
        ref_norm = modality.structural_map(ref_norm)
        ref_resampled = (
            ref_norm if resampled_is_reference else modality.structural_map(ref_resampled)
        )
    elif representation != "intensity":
        raise ValueError(
            f"unknown representation {representation!r}; expected 'intensity' or 'structural'"
        )

    return {
        "src_norm": src_norm,
        "ref_norm": ref_norm,
        "ref_resampled": ref_resampled,
        "representation": representation,
        "requested_ratio": round(ratio, 6),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 2: match
# ---------------------------------------------------------------------------
# LoFTR attention cost grows with the square of the token count, so runtime
# scales roughly with area. Capping the long edge keeps large frames usable;
# matches are found at this resolution and scaled back up, and stage 5 recovers
# sub-pixel precision at full resolution afterwards.
LOFTR_MAX_EDGE = 640


def _match_loftr(model, src_norm, ref_norm, min_confidence, min_matches):
    """Run LoFTR. Returns (pts_src, pts_ref, confidences) or None to fall back."""
    import torch

    device = next(model.parameters()).device

    h_src, w_src = src_norm.shape[:2]
    h_ref, w_ref = ref_norm.shape[:2]

    # LoFTR requires both inputs at a common size divisible by 8.
    scale = min(1.0, LOFTR_MAX_EDGE / max(w_src, h_src))
    target_w = max(8, (int(w_src * scale) // 8) * 8)
    target_h = max(8, (int(h_src * scale) // 8) * 8)

    t_src = cv2.resize(src_norm, (target_w, target_h))
    t_ref = cv2.resize(ref_norm, (target_w, target_h))

    src_t = torch.from_numpy(t_src).float()[None, None].to(device) / 255.0
    ref_t = torch.from_numpy(t_ref).float()[None, None].to(device) / 255.0

    with torch.no_grad():
        correspondences = model({"image0": src_t, "image1": ref_t})

    pts_src = correspondences["keypoints0"].cpu().numpy()
    pts_ref = correspondences["keypoints1"].cpu().numpy()
    confs = correspondences["confidence"].cpu().numpy()

    # Undo the resize independently per image. Using the source's scale factors
    # for both sets would pin the reference points inside the source's extent
    # whenever the two images differ in size.
    pts_src[:, 0] *= w_src / target_w
    pts_src[:, 1] *= h_src / target_h
    pts_ref[:, 0] *= w_ref / target_w
    pts_ref[:, 1] *= h_ref / target_h

    keep = confs >= min_confidence
    if int(np.sum(keep)) < min_matches:
        return None

    return pts_src[keep], pts_ref[keep], confs[keep].tolist()


def _match_sift(src_norm, ref_norm, ratio_threshold=0.78):
    sift = cv2.SIFT_create(nfeatures=2500)
    kp1, des1 = sift.detectAndCompute(src_norm, None)
    kp2, des2 = sift.detectAndCompute(ref_norm, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    pts_src, pts_ref, confs = [], [], []
    for pair in raw_matches:
        # knnMatch returns fewer than k neighbours when the train set is small.
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            pts_src.append(kp1[m.queryIdx].pt)
            pts_ref.append(kp2[m.trainIdx].pt)
            confs.append(1.0 - (m.distance / (n.distance + 1e-6)))

    if not pts_src:
        return None
    return np.array(pts_src), np.array(pts_ref), confs


def _match_binary(detector, src_norm, ref_norm, descriptor_bits, ratio_threshold=0.80):
    """Shared path for binary-descriptor detectors (ORB, AKAZE) over Hamming distance."""
    kp1, des1 = detector.detectAndCompute(src_norm, None)
    kp2, des2 = detector.detectAndCompute(ref_norm, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    pts_src, pts_ref, confs = [], [], []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            pts_src.append(kp1[m.queryIdx].pt)
            pts_ref.append(kp2[m.trainIdx].pt)
            # Hamming distance is bounded by the descriptor length, so it maps
            # onto a 0-1 confidence directly.
            confs.append(max(0.0, 1.0 - m.distance / descriptor_bits))

    if not pts_src:
        return None
    return np.array(pts_src), np.array(pts_ref), confs


def _match_orb(src_norm, ref_norm):
    return _match_binary(cv2.ORB_create(nfeatures=3000), src_norm, ref_norm, 256.0)


def _match_akaze(src_norm, ref_norm):
    return _match_binary(cv2.AKAZE_create(), src_norm, ref_norm, 486.0)


def _match_brisk(src_norm, ref_norm):
    return _match_binary(cv2.BRISK_create(), src_norm, ref_norm, 512.0)


def _detector_available(name):
    """
    OpenCV 5 dropped AKAZE, KAZE and BRISK from the default Python bindings;
    they return only with opencv-contrib. Probe rather than assume, so the API
    advertises the matchers this install can actually run.
    """
    factory = getattr(cv2, f"{name}_create", None)
    if factory is None:
        return False
    try:
        factory()
        return True
    except Exception:
        return False


_CLASSICAL = {
    "sift": ("SIFT", _match_sift),
    "orb": ("ORB", _match_orb),
}

if _detector_available("AKAZE"):
    _CLASSICAL["akaze"] = ("AKAZE", _match_akaze)
if _detector_available("BRISK"):
    _CLASSICAL["brisk"] = ("BRISK", _match_brisk)

MATCHER_INFO = {
    "auto": {
        "name": "Auto",
        "family": "Hybrid",
        "detail": "LoFTR, falling back to SIFT then ORB",
    },
    "loftr": {
        "name": "LoFTR",
        "family": "Learned",
        "detail": "Detector-free coarse-to-fine transformer",
    },
    "sift": {
        "name": "SIFT",
        "family": "Classical",
        "detail": "128-D float descriptors, FLANN + Lowe ratio",
    },
    "orb": {
        "name": "ORB",
        "family": "Classical",
        "detail": "256-bit binary descriptors, Hamming k-NN",
    },
    "akaze": {
        "name": "AKAZE",
        "family": "Classical",
        "detail": "486-bit M-LDB descriptors, Hamming k-NN",
    },
    "brisk": {
        "name": "BRISK",
        "family": "Classical",
        "detail": "512-bit binary descriptors, Hamming k-NN",
    },
}

MATCHERS = {"auto", "loftr", *_CLASSICAL}


def available_matchers():
    """Matcher ids this install can actually run, with display metadata."""
    return [
        {"id": key, **MATCHER_INFO[key]}
        for key in ("auto", "loftr", "sift", "orb", "akaze", "brisk")
        if key in MATCHERS
    ]


def stage2_match(src_norm, ref_norm, matcher="auto", min_confidence=0.55, min_matches=15):
    """
    Find correspondences with the requested matcher.

    "auto" tries LoFTR and falls back to SIFT then ORB if it is unavailable or
    returns too few confident matches. Naming a matcher explicitly runs only
    that one, so a comparison between matchers measures the matcher rather than
    the fallback chain.
    """
    t0 = time.time()

    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; expected one of {sorted(MATCHERS)}")

    result = None
    is_fallback = False

    if matcher in _CLASSICAL:
        method, fn = _CLASSICAL[matcher]
        result = fn(src_norm, ref_norm)
    else:
        method = "LoFTR"
        model = get_loftr_model()
        if model is not None:
            try:
                result = _match_loftr(model, src_norm, ref_norm, min_confidence, min_matches)
                if result is None:
                    print("[pipeline] LoFTR returned too few confident matches")
            except Exception as exc:
                print(f"[pipeline] LoFTR inference failed: {exc}")
                result = None

        # Only "auto" is allowed to substitute a different matcher. An explicit
        # "loftr" request that fails must report that, not quietly return SIFT
        # results under LoFTR's name.
        if result is None and matcher == "auto":
            is_fallback = True
            for key in ("sift", "orb"):
                name, fn = _CLASSICAL[key]
                result = fn(src_norm, ref_norm)
                if result is not None:
                    method = f"{name} (fallback)"
                    break

    if result is None:
        pts_src, pts_ref, confs = np.empty((0, 2)), np.empty((0, 2)), []
    else:
        pts_src, pts_ref, confs = result

    return {
        "method": method,
        "matcher": matcher,
        "is_fallback": is_fallback,
        "pts_src": np.asarray(pts_src, dtype=np.float64),
        "pts_ref": np.asarray(pts_ref, dtype=np.float64),
        "match_count": len(pts_src),
        "confidence": round(float(np.mean(confs)) * 100, 1) if confs else 0.0,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 3: verify
# ---------------------------------------------------------------------------

# A homography has 8 degrees of freedom, so any 4 correspondences fit one
# exactly and report zero residual regardless of whether they describe the same
# scene. Demand a real margin over that minimum before trusting a fit.
MIN_INLIERS = 12
MIN_INLIER_RATIO = 10.0


def is_plausible_homography(H, width, height, max_scale=20.0):
    """
    Reject fits that map the source outside anything physically sensible.

    RANSAC on mismatched images can still return a matrix: typically one that
    folds, mirrors or explodes the frame. Projecting the source corners and
    checking the result is still a sane convex quadrilateral catches those.
    """
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        return False

    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    ).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False

    if not np.all(np.isfinite(projected)):
        return False

    # Turn direction at each vertex must keep a consistent sign for a convex,
    # non-self-intersecting quad. A sign flip means the frame folded over.
    # (numpy 2 dropped 2D np.cross, so take the z component directly.)
    edges = np.roll(projected, -1, axis=0) - projected
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return False

    # And the projected area must stay within a broad factor of the original.
    x, y = projected[:, 0], projected[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    original_area = float(width * height)
    if original_area <= 0:
        return False

    ratio = area / original_area
    return 1.0 / max_scale**2 < ratio < max_scale**2


def stage3_verify(pts_src, pts_ref, reproj_threshold=3.0, frame_size=None):
    """Estimate the homography with MAGSAC++ and drop correspondences it rejects."""
    t0 = time.time()

    if len(pts_src) < MIN_INLIERS:
        raise ValueError(
            f"Only {len(pts_src)} correspondences found; "
            f"at least {MIN_INLIERS} are needed for a trustworthy fit"
        )

    H, mask = cv2.findHomography(
        pts_src,
        pts_ref,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=2000,
        confidence=0.999,
    )

    if H is None or mask is None:
        raise ValueError("MAGSAC++ could not fit a homography to these correspondences")

    inlier_mask = mask.ravel() == 1
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = round((inlier_count / len(pts_src)) * 100, 1)

    if inlier_count < MIN_INLIERS:
        raise ValueError(
            f"Only {inlier_count} of {len(pts_src)} correspondences survived verification; "
            f"at least {MIN_INLIERS} are needed. The images may not overlap"
        )

    if inlier_ratio < MIN_INLIER_RATIO:
        raise ValueError(
            f"Only {inlier_ratio}% of correspondences agree on a single transform; "
            "the images do not appear to show the same scene"
        )

    if frame_size is not None and not is_plausible_homography(H, *frame_size):
        raise ValueError(
            "The estimated transform distorts the source beyond anything physically "
            "plausible; the images do not appear to show the same scene"
        )

    return {
        "H": H,
        "inliers_src": pts_src[inlier_mask],
        "inliers_ref": pts_ref[inlier_mask],
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 4: align
# ---------------------------------------------------------------------------
def stage4_align(src_img, ref_img, H):
    """Warp the source into the reference coordinate frame."""
    t0 = time.time()

    h_ref, w_ref = ref_img.shape[:2]
    warped = cv2.warpPerspective(
        src_img,
        H,
        (w_ref, h_ref),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return {"warped_src": warped, "duration_ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# Stage 5: refine
# ---------------------------------------------------------------------------
def _subpixel_peak(response, peak_x, peak_y):
    """Fit a 1D parabola through the correlation peak on each axis."""
    dx = dy = 0.0

    if 0 < peak_x < response.shape[1] - 1:
        grad = (response[peak_y, peak_x + 1] - response[peak_y, peak_x - 1]) / 2.0
        curv = (
            response[peak_y, peak_x + 1]
            - 2 * response[peak_y, peak_x]
            + response[peak_y, peak_x - 1]
        )
        if abs(curv) > 1e-5:
            dx = -grad / curv

    if 0 < peak_y < response.shape[0] - 1:
        grad = (response[peak_y + 1, peak_x] - response[peak_y - 1, peak_x]) / 2.0
        curv = (
            response[peak_y + 1, peak_x]
            - 2 * response[peak_y, peak_x]
            + response[peak_y - 1, peak_x]
        )
        if abs(curv) > 1e-5:
            dy = -grad / curv

    # A parabola fit only makes sense within one sample of the peak.
    return float(np.clip(dx, -1.0, 1.0)), float(np.clip(dy, -1.0, 1.0))


def _mi_surface(template, search, search_radius):
    """
    Mutual information between the template and every offset in the search
    window, laid out like a matchTemplate response so the same peak-finding and
    parabola fit apply.

    Slower than NCC — there is no FFT shortcut for a joint histogram — but it is
    the measure that survives a change of modality, where NCC does not.
    """
    import modality

    size = 2 * search_radius + 1
    surface = np.zeros((size, size), dtype=np.float32)
    th, tw = template.shape

    for dy in range(size):
        for dx in range(size):
            window = search[dy : dy + th, dx : dx + tw]
            surface[dy, dx] = modality.mutual_information(template, window)

    return surface


def stage5_refine(
    src_gray,
    ref_gray,
    inliers_src,
    inliers_ref,
    H,
    patch_size=15,
    search_radius=4,
    ncc_threshold=0.65,
    similarity="ncc",
    max_points=None,
):
    """
    Re-localise each verified correspondence to sub-pixel precision with NCC
    template matching, then refit the homography on the refined points.

    Both point sets must already be in full-resolution image coordinates.
    """
    t0 = time.time()

    half = patch_size // 2
    h_src, w_src = src_gray.shape[:2]
    h_ref, w_ref = ref_gray.shape[:2]

    refined_src, refined_ref = [], []

    # MI costs roughly three orders of magnitude more per point than NCC, so
    # cap how many correspondences are refined when it is in use. The homography
    # is over-determined many times over by a few hundred points.
    pairs = list(zip(inliers_src, inliers_ref))
    if max_points and len(pairs) > max_points:
        pairs = pairs[:: max(1, len(pairs) // max_points)]

    for p_src, p_ref in pairs:
        x1, y1 = int(round(p_src[0])), int(round(p_src[1]))
        x2, y2 = int(round(p_ref[0])), int(round(p_ref[1]))

        margin = half + search_radius
        if (
            x1 - half < 0 or x1 + half >= w_src or y1 - half < 0 or y1 + half >= h_src
            or x2 - margin < 0 or x2 + margin >= w_ref
            or y2 - margin < 0 or y2 + margin >= h_ref
        ):
            continue

        template = src_gray[y1 - half : y1 + half + 1, x1 - half : x1 + half + 1]
        search = ref_gray[y2 - margin : y2 + margin + 1, x2 - margin : x2 + margin + 1]

        if similarity == "mi":
            response = _mi_surface(template, search, search_radius)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            # MI is in bits, not a correlation, so the NCC threshold does not
            # apply. Accept any peak that stands clear of the window mean.
            if max_val <= response.mean() + 1e-6:
                continue
        else:
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if max_val < ncc_threshold:
                continue

        peak_x, peak_y = max_loc
        dx, dy = _subpixel_peak(response, peak_x, peak_y)

        refined_src.append([float(p_src[0]), float(p_src[1])])
        refined_ref.append(
            [(x2 - search_radius) + peak_x + dx, (y2 - search_radius) + peak_y + dy]
        )

    refined_count = len(refined_src)

    if refined_count >= 4:
        refined_H, _ = cv2.findHomography(
            np.array(refined_src),
            np.array(refined_ref),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=1.5,
        )
        final_H = refined_H if refined_H is not None else H
    else:
        final_H = H

    residual_rmse = reprojection_rmse(final_H, inliers_src, inliers_ref)

    return {
        "final_H": final_H,
        "refined_count": refined_count,
        "residual_rmse": round(residual_rmse, 3),
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def reprojection_errors(H, pts_src, pts_ref):
    """Per-correspondence distance between a reference point and its reprojection."""
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    if len(pts_src) == 0:
        return np.empty(0)

    homogeneous = np.hstack([pts_src, np.ones((len(pts_src), 1))])
    projected = (H @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(pts_ref - projected, axis=1)


def reprojection_rmse(H, pts_src, pts_ref):
    """RMS distance between reference points and their reprojected source points."""
    errors = reprojection_errors(H, pts_src, pts_ref)
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors**2)))


def residual_distribution(errors, bins=24):
    """Summary statistics and a histogram of per-correspondence residuals."""
    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) == 0:
        return None

    upper = float(np.percentile(errors, 99)) or float(errors.max()) or 1.0
    counts, edges = np.histogram(errors, bins=bins, range=(0.0, max(upper, 1e-6)))

    return {
        "rmse_px": round(float(np.sqrt(np.mean(errors**2))), 4),
        "mean_px": round(float(np.mean(errors)), 4),
        "median_px": round(float(np.median(errors)), 4),
        "p95_px": round(float(np.percentile(errors, 95)), 4),
        "max_px": round(float(np.max(errors)), 4),
        "histogram": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in edges],
    }


def spatial_uniformity(points, width, height, grid=6):
    """
    How evenly the verified correspondences cover the frame.

    A homography fitted from matches clustered in one corner extrapolates badly
    across the rest of the image, so coverage matters independently of count.
    Score combines the fraction of occupied cells with how even the occupied
    counts are; 1.0 is perfectly uniform, 0.0 is a single cluster.
    """
    cells = np.zeros(grid * grid, dtype=int)
    points = np.asarray(points, dtype=np.float64)

    if len(points) == 0 or width <= 0 or height <= 0:
        return {"grid": grid, "occupancy": cells.tolist(), "occupied_cells": 0, "score": 0.0}

    gx = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
    np.add.at(cells, gy * grid + gx, 1)

    occupied = int(np.count_nonzero(cells))
    coverage = occupied / cells.size

    # Coefficient of variation over occupied cells only, so a well-spread but
    # sparse set is not punished twice for the empty cells.
    filled = cells[cells > 0]
    cv = float(np.std(filled) / np.mean(filled)) if len(filled) else 1.0
    evenness = 1.0 / (1.0 + cv)

    return {
        "grid": grid,
        "occupancy": cells.tolist(),
        "occupied_cells": occupied,
        "coverage": round(coverage, 4),
        "score": round(float(coverage * evenness), 4),
    }


def decompose_homography(H):
    """
    Break the homography into interpretable parts.

    The upper-left 2x2 is factored by QR into a rotation times an upper
    triangular matrix, giving separate x/y scales and a shear term. Values are
    exact for a similarity or affine transform and approximate once the
    projective row is significant, so that row is reported too.
    """
    H = np.asarray(H, dtype=np.float64)
    A = H[:2, :2]

    Q, R = np.linalg.qr(A)

    # QR is only unique up to sign; force positive diagonal so the rotation is
    # a true rotation rather than a rotation composed with a reflection.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    R = R * signs[:, None]

    scale_x, scale_y = float(R[0, 0]), float(R[1, 1])
    shear = float(R[0, 1] / R[1, 1]) if abs(R[1, 1]) > 1e-12 else 0.0
    rotation = float(np.degrees(np.arctan2(Q[1, 0], Q[0, 0])))

    return {
        "scale": round(float(np.sqrt(abs(scale_x * scale_y))), 6),
        "scale_x": round(scale_x, 6),
        "scale_y": round(scale_y, 6),
        "rotation_deg": round(rotation, 4),
        "shear": round(shear, 6),
        "tx_px": round(float(H[0, 2]), 3),
        "ty_px": round(float(H[1, 2]), 3),
        "perspective": [float(H[2, 0]), float(H[2, 1])],
        "reflected": bool(np.linalg.det(A) < 0),
    }


def corner_error(H_estimated, H_truth, width, height):
    """
    Mean corner displacement between an estimated homography and a known one.

    This is an independent accuracy measure: unlike the reprojection residual it
    cannot be driven down by the estimator agreeing with its own inliers. Only
    available for pairs with a recorded ground-truth transform.
    """
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    projected_est = cv2.perspectiveTransform(corners, H_estimated).reshape(-1, 2)
    projected_truth = cv2.perspectiveTransform(corners, H_truth).reshape(-1, 2)

    distances = np.linalg.norm(projected_est - projected_truth, axis=1)
    return {
        "mean_corner_error_px": round(float(np.mean(distances)), 3),
        "max_corner_error_px": round(float(np.max(distances)), 3),
    }


# ---------------------------------------------------------------------------
# Stage 6: report
# ---------------------------------------------------------------------------
CONDITION_PROFILES = {
    "same_sensor": {
        "label": "Same sensor, small geometric offset",
        "rmse_limit": 1.0,
        "inlier_target": 80.0,
    },
    "cross_sensor": {
        "label": "Cross sensor, resampled to a common GSD",
        "rmse_limit": 1.5,
        "inlier_target": 70.0,
    },
    "sun_angle_delta": {
        "label": "Illumination change between acquisitions",
        "rmse_limit": 1.2,
        "inlier_target": 65.0,
    },
    "cross_modal": {
        "label": "Cross modal, no shared intensity relationship",
        "rmse_limit": 2.0,
        "inlier_target": 55.0,
    },
}


def stage6_report(metrics, condition="same_sensor", ground_truth=None):
    """Score the run against the accuracy budget for its acquisition condition."""
    profile = CONDITION_PROFILES.get(condition, CONDITION_PROFILES["same_sensor"])

    rmse = metrics.get("residual_rmse", float("nan"))
    inlier_ratio = metrics.get("inlier_ratio", 0.0)

    meets_rmse = rmse < profile["rmse_limit"]
    meets_inliers = inlier_ratio >= profile["inlier_target"]

    report = {
        "condition": condition,
        "condition_label": profile["label"],
        "rmse_limit_px": profile["rmse_limit"],
        "inlier_target_pct": profile["inlier_target"],
        "residual_rmse_px": rmse,
        "inlier_count": metrics.get("inlier_count", 0),
        "inlier_ratio_pct": inlier_ratio,
        "confidence_pct": metrics.get("confidence", 0.0),
        "refined_count": metrics.get("refined_count", 0),
        "method_used": metrics.get("method", "unknown"),
        "subpixel": bool(rmse < 1.0),
        "subpixel_status": (
            "Sub-pixel (residual < 1.0 px)" if rmse < 1.0 else "Pixel-level (residual >= 1.0 px)"
        ),
        "meets_rmse_budget": bool(meets_rmse),
        "meets_inlier_budget": bool(meets_inliers),
        "evaluation_grade": "OPTIMAL" if (meets_rmse and meets_inliers) else "ACCEPTABLE",
    }

    if ground_truth is not None:
        report["ground_truth"] = ground_truth
        report["verified_against_ground_truth"] = True
    else:
        report["verified_against_ground_truth"] = False

    return report


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def execute_pipeline(
    src_img,
    ref_img,
    condition="same_sensor",
    src_gsd=1.0,
    ref_gsd=1.0,
    ground_truth_H=None,
    matcher="auto",
    ransac_threshold=3.0,
    use_clahe=True,
    gamma=1.0,
    denoise=0.0,
    representation="intensity",
    similarity="ncc",
):
    """
    Run all six stages and return a dict keyed ``stage1`` .. ``stage6``.

    Raises ValueError when matching or verification cannot produce a homography.
    """
    results = {}

    s1 = stage1_normalize(
        src_img,
        ref_img,
        src_gsd,
        ref_gsd,
        use_clahe=use_clahe,
        gamma=gamma,
        denoise=denoise,
        representation=representation,
    )
    results["stage1"] = s1

    # Stages 2, 3 and 5 all work in the common frame: the source at its native
    # resolution against the reference resampled to the same GSD. NCC in
    # particular is only meaningful when both patches cover the same ground
    # area, so refinement must happen here rather than against the full-size
    # reference.
    s2 = stage2_match(s1["src_norm"], s1["ref_resampled"], matcher=matcher)
    results["stage2"] = s2

    h_src, w_src = s1["src_norm"].shape[:2]
    s3 = stage3_verify(
        s2["pts_src"],
        s2["pts_ref"],
        reproj_threshold=ransac_threshold,
        frame_size=(w_src, h_src),
    )
    results["stage3"] = s3

    s5 = stage5_refine(
        s1["src_norm"],
        s1["ref_resampled"],
        s3["inliers_src"],
        s3["inliers_ref"],
        s3["H"],
        similarity=similarity,
        max_points=400 if similarity == "mi" else None,
    )

    # Map out of the common frame into full-resolution reference pixels. The
    # delivered warp, the reported matrix and every error figure are all in the
    # reference frame the product is actually consumed in.
    to_full = np.diag([1.0 / s1["scale_x"], 1.0 / s1["scale_y"], 1.0])
    final_H = to_full @ s5["final_H"]

    inliers_ref_full = s3["inliers_ref"].copy()
    if len(inliers_ref_full):
        inliers_ref_full[:, 0] /= s1["scale_x"]
        inliers_ref_full[:, 1] /= s1["scale_y"]

    errors = reprojection_errors(final_H, s3["inliers_src"], inliers_ref_full)

    s5["final_H"] = final_H
    s5["residual_rmse"] = round(float(np.sqrt(np.mean(errors**2))), 3) if len(errors) else 0.0
    results["stage5"] = s5

    s3["H"] = to_full @ s3["H"]
    s3["inliers_ref"] = inliers_ref_full

    results["stage4"] = stage4_align(src_img, ref_img, final_H)

    results["residuals"] = residual_distribution(errors)
    results["uniformity"] = spatial_uniformity(s3["inliers_src"], w_src, h_src)
    results["transform"] = decompose_homography(final_H)

    ground_truth = None
    if ground_truth_H is not None:
        truth_H = np.asarray(ground_truth_H, dtype=np.float64)
        ground_truth = corner_error(final_H, truth_H, w_src, h_src)
        # Decompose the true matrix with the same routine as the estimate, so the
        # two are directly comparable. Quoting the parameters that were fed to
        # cv2.getRotationMatrix2D instead would invite a sign-convention
        # mismatch that looks like an error but is not one.
        ground_truth["transform"] = decompose_homography(truth_H)

    s6 = stage6_report(
        {
            "residual_rmse": s5["residual_rmse"],
            "inlier_count": s3["inlier_count"],
            "inlier_ratio": s3["inlier_ratio"],
            "confidence": s2["confidence"],
            "refined_count": s5["refined_count"],
            "method": s2["method"],
        },
        condition=condition,
        ground_truth=ground_truth,
    )
    results["stage6"] = s6

    # Overlay coordinates: source in source pixels, reference in full-resolution
    # reference pixels, so the viewer can scale each to its own displayed image.
    results["keypoints"] = [
        {
            "src": {"x": float(p_src[0]), "y": float(p_src[1])},
            "ref": {"x": float(p_ref[0]), "y": float(p_ref[1])},
        }
        for p_src, p_ref in zip(s3["inliers_src"], s3["inliers_ref"])
    ]

    return results


