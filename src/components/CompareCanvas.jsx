import React, { useEffect, useRef, useState } from 'react';

/**
 * Before-and-after residual view.
 *
 * Left:  source stretched over the reference with no registration at all.
 * Right: the warp the pipeline solved.
 *
 * Differencing raw intensities measures the wrong thing when the two frames
 * were acquired under different illumination — or, ultimately, by different
 * instruments. A pair can be geometrically perfect and still differ by 37 grey
 * levels because one is simply brighter. So by default both panels are locally
 * normalised first: each pixel becomes a z-score against the mean and standard
 * deviation of its own neighbourhood, which removes brightness and contrast
 * differences that vary slowly across the frame and leaves genuine structural
 * misalignment behind. "Raw" shows the unnormalised difference for comparison.
 */
const NORM_RADIUS = 12;
const EPSILON = 1e-3;

function luminance(img, width, height) {
  const scratch = document.createElement('canvas');
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);

  const lum = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    lum[p] = (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  return lum;
}

/**
 * Local z-score via summed-area tables, so the window mean and variance cost
 * O(1) per pixel regardless of radius.
 */
function localNormalize(lum, width, height, radius) {
  const stride = width + 1;
  const sum = new Float64Array(stride * (height + 1));
  const sumSq = new Float64Array(stride * (height + 1));

  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    let rowSumSq = 0;
    for (let x = 0; x < width; x++) {
      const v = lum[y * width + x];
      rowSum += v;
      rowSumSq += v * v;
      sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + rowSum;
      sumSq[(y + 1) * stride + x + 1] = sumSq[y * stride + x + 1] + rowSumSq;
    }
  }

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height, y + radius + 1);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width, x + radius + 1);
      const n = (y1 - y0) * (x1 - x0);

      const a = y1 * stride + x1;
      const b = y0 * stride + x1;
      const c = y1 * stride + x0;
      const d = y0 * stride + x0;

      const mean = (sum[a] - sum[b] - sum[c] + sum[d]) / n;
      const meanSq = (sumSq[a] - sumSq[b] - sumSq[c] + sumSq[d]) / n;
      const variance = Math.max(0, meanSq - mean * mean);

      out[y * width + x] = (lum[y * width + x] - mean) / (Math.sqrt(variance) + EPSILON);
    }
  }
  return out;
}

function drawDifference(ctx, baseImg, overlayImg, width, height, gain, normalize) {
  const lumA = luminance(baseImg, width, height);
  const lumB = luminance(overlayImg, width, height);

  const a = normalize ? localNormalize(lumA, width, height, NORM_RADIUS) : lumA;
  const b = normalize ? localNormalize(lumB, width, height, NORM_RADIUS) : lumB;

  // Normalised values are roughly unit variance, so they need a much larger
  // display multiplier than raw grey levels to occupy the same visual range.
  const displayScale = normalize ? 24 : 1;

  const out = ctx.createImageData(width, height);
  let total = 0;
  let counted = 0;

  for (let p = 0; p < a.length; p++) {
    const diff = Math.abs(a[p] - b[p]);

    // The warp leaves zeroed borders where the source has no data. Counting
    // those would flatter the unregistered panel, so skip pixels that are
    // black in either original frame.
    if (lumA[p] > 2 && lumB[p] > 2) {
      total += diff;
      counted++;
    }

    const shown = Math.min(255, diff * displayScale * gain);
    const i = p * 4;
    out.data[i] = shown;
    out.data[i + 1] = shown;
    out.data[i + 2] = shown;
    out.data[i + 3] = 255;
  }

  ctx.putImageData(out, 0, 0);
  return counted ? total / counted : 0;
}

export default function CompareCanvas({ sourceImg, referenceImg, warpedImg, gain, normalize }) {
  const beforeRef = useRef(null);
  const afterRef = useRef(null);
  const [stats, setStats] = useState({ before: 0, after: 0 });

  useEffect(() => {
    if (!sourceImg || !referenceImg || !warpedImg) return;

    const width = referenceImg.naturalWidth;
    const height = referenceImg.naturalHeight;

    for (const ref of [beforeRef, afterRef]) {
      if (!ref.current) return;
      ref.current.width = width;
      ref.current.height = height;
    }

    const before = drawDifference(
      beforeRef.current.getContext('2d'),
      referenceImg,
      sourceImg,
      width,
      height,
      gain,
      normalize
    );
    const after = drawDifference(
      afterRef.current.getContext('2d'),
      referenceImg,
      warpedImg,
      width,
      height,
      gain,
      normalize
    );

    setStats({ before, after });
  }, [sourceImg, referenceImg, warpedImg, gain, normalize]);

  const improvement = stats.after > 0 ? stats.before / stats.after : 0;
  const unit = normalize ? 'σ' : 'levels';
  const digits = normalize ? 3 : 2;

  return (
    <div className="compare-wrap">
      <figure className="compare-pane">
        <canvas ref={beforeRef} />
        <figcaption>
          <span className="compare-label">Unregistered</span>
          <span className="mono val-amber">
            {stats.before.toFixed(digits)} {unit}
          </span>
        </figcaption>
      </figure>

      <figure className="compare-pane">
        <canvas ref={afterRef} />
        <figcaption>
          <span className="compare-label">Registered</span>
          <span className="mono val-cyan">
            {stats.after.toFixed(digits)} {unit}
          </span>
        </figcaption>
      </figure>

      <div className="compare-verdict mono">
        {improvement > 1 && (
          <>
            structure mismatch reduced <strong>{improvement.toFixed(1)}×</strong> ·{' '}
          </>
        )}
        {normalize
          ? 'locally normalised, so illumination differences are excluded'
          : 'raw intensity difference, including any illumination difference'}
        {' · '}amplified ×{gain}
      </div>
    </div>
  );
}
