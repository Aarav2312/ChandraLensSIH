import asyncio
import base64
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MANIFEST_PATH = os.path.join(DATASETS_DIR, "manifest.json")
DIST_DIR = os.path.join(BASE_DIR, "dist")

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
CACHE_SIZE = 24

# LoFTR can return several thousand correspondences. The overlay draws a subset
# anyway, so shipping them all just inflates the response; send an evenly spaced
# sample and report the true count alongside it.
MAX_KEYPOINTS_SENT = 600

# Results are deterministic for a given input and parameter set, so repeating a
# run — flipping between tabs, re-running to show someone — can be served from
# memory instead of paying for LoFTR again.
_result_cache = OrderedDict()
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0}


@asynccontextmanager
async def lifespan(_app):
    # Load LoFTR during startup rather than on the first request, so the first
    # registration a user runs is not several seconds slower than the rest.
    asyncio.get_running_loop().run_in_executor(None, pipeline.get_loftr_model)
    yield
    _result_cache.clear()


app = FastAPI(title="Registration API", version="4.0.0", lifespan=lifespan)

# The frontend is served from this same origin in production; CORS only matters
# for the Vite dev server on :5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_manifest():
    """Read the dataset manifest fresh so regenerating it needs no restart."""
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def public_pair(entry):
    """The subset of a manifest entry the frontend needs. Ground truth stays server-side."""
    return {
        "id": entry["id"],
        "name": entry["name"],
        "summary": entry["summary"],
        "condition": entry["condition"],
        "sourceUrl": entry["source_url"],
        "referenceUrl": entry["reference_url"],
        "sourceSize": entry["source_size"],
        "referenceSize": entry["reference_size"],
        "gsdRatio": entry["gsd_ratio"],
        "synthetic": entry.get("synthetic", False),
        "appliedTransform": entry.get("applied_transform"),
        "provenance": entry.get("provenance"),
    }


def encode_image(image):
    """
    PNG-encode for the browser, dropping to a single channel when the three are
    identical. Lunar frames are greyscale stored as RGB, so this cuts the
    payload by roughly two thirds. PNG rather than JPEG because the compare view
    subtracts these images and JPEG ringing would show up as false residual.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        if np.array_equal(b, g) and np.array_equal(g, r):
            image = b

    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode the warped image")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def cache_get(key):
    with _cache_lock:
        if key in _result_cache:
            _result_cache.move_to_end(key)
            _cache_stats["hits"] += 1
            return _result_cache[key]
        _cache_stats["misses"] += 1
        return None


def cache_put(key, value):
    with _cache_lock:
        _result_cache[key] = value
        _result_cache.move_to_end(key)
        while len(_result_cache) > CACHE_SIZE:
            _result_cache.popitem(last=False)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "loftr_loaded": pipeline._LOFTR_MODEL is not None,
        "opencv": cv2.__version__,
        "pairs_available": len(load_manifest()),
        "matchers": pipeline.available_matchers(),
        "cache": dict(_cache_stats, size=len(_result_cache)),
    }


@app.get("/api/samples")
def samples():
    return {
        "pairs": [public_pair(entry) for entry in load_manifest()],
        "matchers": pipeline.available_matchers(),
        "conditions": [
            {"id": key, "label": profile["label"], **profile}
            for key, profile in pipeline.CONDITION_PROFILES.items()
        ],
    }


def decode_upload(raw_bytes, label):
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} image exceeds 32 MB")
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"could not decode the {label} image")
    return image


def run_pipeline_sync(src_img, ref_img, options, ground_truth_H):
    """CPU-bound work, called off the event loop."""
    started = time.time()
    results = pipeline.execute_pipeline(
        src_img, ref_img, ground_truth_H=ground_truth_H, **options
    )
    elapsed_ms = round((time.time() - started) * 1000, 1)

    s2, s3, s5, s6 = (results["stage" + n] for n in "2356")

    keypoints = results["keypoints"]
    step = max(1, len(keypoints) // MAX_KEYPOINTS_SENT)

    return {
        "status": "success",
        "method": s2["method"],
        "matcher": s2["matcher"],
        "isFallback": s2["is_fallback"],
        "matchCount": s2["match_count"],
        "confidence": s2["confidence"],
        "inlierCount": s3["inlier_count"],
        "inlierRatio": s3["inlier_ratio"],
        "residualRmse": s5["residual_rmse"],
        "homographyMatrix": s5["final_H"].tolist(),
        "transform": results["transform"],
        "residuals": results["residuals"],
        "uniformity": results["uniformity"],
        "keypoints": keypoints[::step],
        "keypointsSampled": step > 1,
        "report": s6,
        "warpedDataUrl": encode_image(results["stage4"]["warped_src"]),
        "processingTimeMs": elapsed_ms,
        "options": options,
        "timings": {
            "NORMALIZE": results["stage1"]["duration_ms"],
            "MATCH": s2["duration_ms"],
            "VERIFY": s3["duration_ms"],
            "ALIGN": results["stage4"]["duration_ms"],
            "REFINE": s5["duration_ms"],
            "REPORT": 0.0,
        },
    }


@app.post("/api/register")
async def register(
    pair_id: str = Form(None),
    condition: str = Form("same_sensor"),
    src_gsd: float = Form(1.0),
    ref_gsd: float = Form(1.0),
    matcher: str = Form("auto"),
    ransac_threshold: float = Form(3.0),
    use_clahe: bool = Form(True),
    gamma: float = Form(1.0),
    denoise: float = Form(0.0),
    representation: str = Form("intensity"),
    similarity: str = Form("ncc"),
    src_file: UploadFile = File(None),
    ref_file: UploadFile = File(None),
):
    if matcher not in pipeline.MATCHERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown matcher {matcher!r}; available: {sorted(pipeline.MATCHERS)}",
        )
    if not 0.5 <= ransac_threshold <= 20.0:
        raise HTTPException(status_code=400, detail="ransac_threshold must be between 0.5 and 20")
    if not 0.2 <= gamma <= 3.0:
        raise HTTPException(status_code=400, detail="gamma must be between 0.2 and 3.0")
    if representation not in ("intensity", "structural"):
        raise HTTPException(status_code=400, detail="representation must be intensity or structural")
    if similarity not in ("ncc", "mi"):
        raise HTTPException(status_code=400, detail="similarity must be ncc or mi")

    ground_truth_H = None
    cache_key = None

    if src_file is not None and ref_file is not None:
        src_bytes = await src_file.read()
        ref_bytes = await ref_file.read()
        src_img = decode_upload(src_bytes, "source")
        ref_img = decode_upload(ref_bytes, "reference")
        # Hash the uploads so re-running the same files with the same settings
        # still hits the cache.
        digest = hashlib.sha256(src_bytes).hexdigest()[:16]
        digest += hashlib.sha256(ref_bytes).hexdigest()[:16]
        cache_key = ("upload", digest)
    elif pair_id:
        entry = next((p for p in load_manifest() if p["id"] == pair_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown pair: {pair_id}")

        src_img = cv2.imread(os.path.join(BASE_DIR, entry["source_path"]))
        ref_img = cv2.imread(os.path.join(BASE_DIR, entry["reference_path"]))
        if src_img is None or ref_img is None:
            raise HTTPException(
                status_code=500,
                detail=f"dataset files for {pair_id} are missing; run prepare_datasets.py",
            )

        # Manifest values win over anything the client sent.
        condition = entry["condition"]
        src_gsd = entry["src_gsd"]
        ref_gsd = entry["ref_gsd"]
        ground_truth_H = entry.get("ground_truth_H")
        cache_key = ("pair", pair_id)
    else:
        raise HTTPException(status_code=400, detail="provide either pair_id or both image files")

    options = {
        "condition": condition,
        "src_gsd": src_gsd,
        "ref_gsd": ref_gsd,
        "matcher": matcher,
        "ransac_threshold": ransac_threshold,
        "use_clahe": use_clahe,
        "gamma": gamma,
        "denoise": denoise,
        "representation": representation,
        "similarity": similarity,
    }

    cache_key = cache_key + tuple(sorted(options.items()))
    cached = cache_get(cache_key)
    if cached is not None:
        return dict(cached, cached=True)

    try:
        payload = await run_in_threadpool(
            run_pipeline_sync, src_img, ref_img, options, ground_truth_H
        )
    except ValueError as exc:
        # Not enough correspondences to solve. This is an expected outcome for
        # hard pairs, not a server error, so report it as a normal result.
        return JSONResponse(
            status_code=200,
            content={"status": "failed", "failureReason": str(exc), "options": options},
        )

    cache_put(cache_key, payload)
    return dict(payload, cached=False)


if os.path.isdir(DATASETS_DIR):
    app.mount("/datasets", StaticFiles(directory=DATASETS_DIR), name="datasets")

# Mounted last so it does not shadow the API routes above.
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    if not os.path.exists(MANIFEST_PATH):
        print("warning: datasets/manifest.json missing. Run: python prepare_datasets.py")
    uvicorn.run(app, host="0.0.0.0", port=port)
