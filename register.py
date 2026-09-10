"""
Command-line front end to the registration pipeline.

    python register.py --pair cleomedes
    python register.py --all
    python register.py --source a.png --reference b.png --src-gsd 4 --ref-gsd 1

Writes the warped product, a match visualisation, the homography and a JSON
metrics report into the output directory.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "datasets", "manifest.json")

# BGR. Cyan for LoFTR, amber when a fallback matcher produced the result.
COLOR_PRIMARY = (179, 168, 127)
COLOR_FALLBACK = (90, 162, 201)


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def parse_args(manifest_ids):
    parser = argparse.ArgumentParser(
        description="Register a source image onto a reference image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--pair", "-p", choices=manifest_ids, help="Run a pair from datasets/manifest.json"
    )
    selection.add_argument("--all", "-a", action="store_true", help="Run every manifest pair")
    selection.add_argument("--source", "-s", help="Path to the moving source image")

    parser.add_argument("--reference", "-r", help="Path to the fixed reference image")
    parser.add_argument("--output-dir", "-o", default="results", help="Where to write products")
    parser.add_argument(
        "--condition",
        "-c",
        default="same_sensor",
        choices=sorted(pipeline.CONDITION_PROFILES),
        help="Acquisition condition, selects the accuracy budget",
    )
    parser.add_argument("--src-gsd", type=float, default=1.0, help="Source ground sample distance")
    parser.add_argument("--ref-gsd", type=float, default=1.0, help="Reference ground sample distance")
    parser.add_argument(
        "--matcher", "-m", default="auto", choices=sorted(pipeline.MATCHERS), help="Matcher to use"
    )
    parser.add_argument(
        "--representation",
        default="intensity",
        choices=["intensity", "structural"],
        help="Match on raw intensity, or on a phase-congruency structural map for cross-modal pairs",
    )
    parser.add_argument(
        "--similarity",
        default="ncc",
        choices=["ncc", "mi"],
        help="Sub-pixel refinement measure; mi survives a change of modality, ncc does not",
    )
    parser.add_argument("--src-band", type=int, help="Band index for a PDS4 spectral cube source")
    parser.add_argument("--ref-band", type=int, help="Band index for a PDS4 spectral cube reference")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress the per-run summary")

    args = parser.parse_args()
    if args.source and not args.reference:
        parser.error("--source requires --reference")
    return args


# A dense matcher returns thousands of correspondences spread evenly over the
# frame. Drawn all at once they collapse into a solid hatch that shows nothing.
MAX_DRAWN_MATCHES = 120
LINE_OPACITY = 0.5


def draw_matches(src_img, ref_img, keypoints, is_fallback):
    """Side-by-side source and reference with a line per sampled correspondence."""
    h_src, w_src = src_img.shape[:2]
    h_ref, w_ref = ref_img.shape[:2]

    canvas = np.zeros((max(h_src, h_ref), w_src + w_ref, 3), dtype=np.uint8)
    canvas[:h_src, :w_src] = src_img
    canvas[:h_ref, w_src : w_src + w_ref] = ref_img

    color = COLOR_FALLBACK if is_fallback else COLOR_PRIMARY
    step = max(1, len(keypoints) // MAX_DRAWN_MATCHES)
    sampled = keypoints[::step]

    # Lines go on their own layer so they can be blended back at partial
    # opacity, leaving the terrain underneath readable.
    lines = canvas.copy()
    for kp in sampled:
        pt_src = (int(kp["src"]["x"]), int(kp["src"]["y"]))
        pt_ref = (int(kp["ref"]["x"]) + w_src, int(kp["ref"]["y"]))
        cv2.line(lines, pt_src, pt_ref, color, 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(lines, LINE_OPACITY, canvas, 1 - LINE_OPACITY, 0)

    for kp in sampled:
        cv2.circle(canvas, (int(kp["src"]["x"]), int(kp["src"]["y"])), 2, color, -1, cv2.LINE_AA)
        cv2.circle(
            canvas, (int(kp["ref"]["x"]) + w_src, int(kp["ref"]["y"])), 2, color, -1, cv2.LINE_AA
        )

    return canvas


def build_report(results, elapsed_ms, meta):
    s2, s3, s5, s6 = (results["stage" + n] for n in "2356")
    return {
        "pair": meta.get("id"),
        "condition": s6["condition"],
        "condition_label": s6["condition_label"],
        "execution_time_ms": elapsed_ms,
        "matcher": {
            "method": s2["method"],
            "is_fallback": s2["is_fallback"],
            "matches_found": s2["match_count"],
            "mean_confidence_pct": s2["confidence"],
        },
        "verification": {
            "inliers": s3["inlier_count"],
            "inlier_ratio_pct": s3["inlier_ratio"],
        },
        "accuracy": {
            "residual_rmse_px": s5["residual_rmse"],
            "subpixel": s6["subpixel"],
            "refined_correspondences": s5["refined_count"],
            "residual_distribution": {
                k: v
                for k, v in (results.get("residuals") or {}).items()
                if k not in ("histogram", "bin_edges")
            },
            "ground_truth": s6.get("ground_truth"),
        },
        "distribution": results.get("uniformity"),
        "recovered_transform": results.get("transform"),
        "budget": {
            "rmse_limit_px": s6["rmse_limit_px"],
            "inlier_target_pct": s6["inlier_target_pct"],
            "meets_rmse_budget": s6["meets_rmse_budget"],
            "meets_inlier_budget": s6["meets_inlier_budget"],
            "evaluation_grade": s6["evaluation_grade"],
        },
        "gsd": {
            "source": meta.get("src_gsd"),
            "reference": meta.get("ref_gsd"),
            "reference_resampled_by": round(results["stage1"]["scale_x"], 6),
        },
        "homography_matrix": s5["final_H"].tolist(),
        "timings_ms": {
            "normalize": results["stage1"]["duration_ms"],
            "match": s2["duration_ms"],
            "verify": s3["duration_ms"],
            "align": results["stage4"]["duration_ms"],
            "refine": s5["duration_ms"],
        },
    }


def print_summary(report):
    acc = report["accuracy"]
    print(f"  matcher            {report['matcher']['method']}")
    print(
        f"  matches / inliers  {report['matcher']['matches_found']} / "
        f"{report['verification']['inliers']} "
        f"({report['verification']['inlier_ratio_pct']}%)"
    )
    print(f"  residual RMSE      {acc['residual_rmse_px']} px (reference frame)")
    dist = report.get("distribution")
    if dist:
        print(
            f"  distribution       {dist['score']} uniformity, "
            f"{dist['occupied_cells']}/{dist['grid'] ** 2} cells occupied"
        )
    if acc["ground_truth"]:
        print(
            f"  vs ground truth    {acc['ground_truth']['mean_corner_error_px']} px mean "
            f"corner error, {acc['ground_truth']['max_corner_error_px']} px max"
        )
    print(
        f"  grade              {report['budget']['evaluation_grade']} "
        f"(budget {report['budget']['rmse_limit_px']} px / "
        f"{report['budget']['inlier_target_pct']}% inliers)"
    )
    print(f"  elapsed            {report['execution_time_ms']} ms")


def load_image(path, band=None):
    """
    Read an ordinary image, or a PDS4 product when handed an XML label.

    Chandrayaan-2 products from ISSDC are a detached XML label plus a raw
    binary; there is no header in the binary, so the label has to be parsed to
    know the shape and element type.
    """
    if path.lower().endswith(".xml"):
        import pds4

        product = pds4.read_product(path, band=band)
        print(f"  {pds4.describe(product)}")
        grey = pds4.to_display_image(product)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    return cv2.imread(path)


def run_one(
    src_path, ref_path, output_dir, condition, src_gsd, ref_gsd, meta, quiet, options=None
):
    src_img = load_image(src_path, (options or {}).get("src_band"))
    ref_img = load_image(ref_path, (options or {}).get("ref_band"))

    if src_img is None:
        raise FileNotFoundError(f"could not read source image: {src_path}")
    if ref_img is None:
        raise FileNotFoundError(f"could not read reference image: {ref_path}")

    os.makedirs(output_dir, exist_ok=True)

    started = time.time()
    results = pipeline.execute_pipeline(
        src_img,
        ref_img,
        condition=condition,
        src_gsd=src_gsd,
        ref_gsd=ref_gsd,
        ground_truth_H=meta.get("ground_truth_H"),
        matcher=(options or {}).get("matcher", "auto"),
        representation=(options or {}).get("representation", "intensity"),
        similarity=(options or {}).get("similarity", "ncc"),
    )
    elapsed_ms = round((time.time() - started) * 1000, 1)

    cv2.imwrite(os.path.join(output_dir, "warped_registered.png"), results["stage4"]["warped_src"])
    cv2.imwrite(
        os.path.join(output_dir, "matches_visualization.png"),
        draw_matches(src_img, ref_img, results["keypoints"], results["stage2"]["is_fallback"]),
    )
    np.savetxt(
        os.path.join(output_dir, "homography_matrix.txt"),
        results["stage5"]["final_H"],
        fmt="%.8f",
        header="3x3 projective homography, source pixels -> reference pixels",
    )

    report = build_report(results, elapsed_ms, {**meta, "src_gsd": src_gsd, "ref_gsd": ref_gsd})
    with open(os.path.join(output_dir, "metrics_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if not quiet:
        print_summary(report)

    return report


def main():
    manifest = load_manifest()
    manifest_by_id = {entry["id"]: entry for entry in manifest}
    args = parse_args(sorted(manifest_by_id))

    if (args.pair or args.all) and not manifest:
        print("No datasets/manifest.json found. Run: python prepare_datasets.py", file=sys.stderr)
        return 1

    if args.all:
        selected = manifest
    elif args.pair:
        selected = [manifest_by_id[args.pair]]
    else:
        selected = None

    options = {
        "matcher": args.matcher,
        "representation": args.representation,
        "similarity": args.similarity,
        "src_band": args.src_band,
        "ref_band": args.ref_band,
    }

    failures = 0

    if selected is None:
        print(f"registering {args.source} -> {args.reference}")
        try:
            run_one(
                args.source,
                args.reference,
                args.output_dir,
                args.condition,
                args.src_gsd,
                args.ref_gsd,
                {},
                args.quiet,
                options,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            return 2
        return 0

    for entry in selected:
        out_dir = args.output_dir if len(selected) == 1 else os.path.join(args.output_dir, entry["id"])
        print(f"\n{entry['id']} - {entry['name']} ({entry['gsd_ratio']} GSD)")
        try:
            run_one(
                os.path.join(BASE_DIR, entry["source_path"]),
                os.path.join(BASE_DIR, entry["reference_path"]),
                out_dir,
                entry["condition"],
                entry["src_gsd"],
                entry["ref_gsd"],
                entry,
                args.quiet,
                options,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
