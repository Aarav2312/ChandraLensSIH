import React, { useEffect, useRef } from 'react';
import { COLORS } from '../constants';

const SWEEP_DURATION_MS = 360;

// LoFTR returns thousands of correspondences spread evenly over the frame.
// Drawn all at once they collapse into a solid hatch that shows nothing, so
// plot an evenly spaced subset; the count in the corner reports the true total.
const MAX_DRAWN = 120;

export default function MatchOverlayCanvas({
  sourceImg,
  referenceImg,
  keypoints,
  isFallback = false,
  animateSweep = false,
  onSweepComplete
}) {
  const canvasRef = useRef(null);
  const frameRef = useRef(null);
  const completeRef = useRef(onSweepComplete);

  completeRef.current = onSweepComplete;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sourceImg || !referenceImg) return undefined;

    const ctx = canvas.getContext('2d');

    // Lay the two frames side by side at their own aspect ratios, scaled to a
    // common height, so neither image is distorted.
    const targetHeight = 640;
    const srcW = Math.round((sourceImg.naturalWidth / sourceImg.naturalHeight) * targetHeight);
    const refW = Math.round((referenceImg.naturalWidth / referenceImg.naturalHeight) * targetHeight);

    canvas.width = srcW + refW;
    canvas.height = targetHeight;

    const scale = {
      srcX: srcW / sourceImg.naturalWidth,
      srcY: targetHeight / sourceImg.naturalHeight,
      refX: refW / referenceImg.naturalWidth,
      refY: targetHeight / referenceImg.naturalHeight
    };

    const points = keypoints?.length
      ? keypoints.filter((_, i) => i % Math.ceil(keypoints.length / MAX_DRAWN) === 0)
      : [];

    const lineColor = isFallback ? COLORS.fallback : COLORS.primary;
    const pointColor = isFallback ? COLORS.fallbackBright : COLORS.primaryBright;

    const drawFrames = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(sourceImg, 0, 0, srcW, targetHeight);
      ctx.drawImage(referenceImg, srcW, 0, refW, targetHeight);

      ctx.strokeStyle = COLORS.hairline;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(srcW + 0.5, 0);
      ctx.lineTo(srcW + 0.5, targetHeight);
      ctx.stroke();

      ctx.font = '12px "IBM Plex Mono", monospace';
      ctx.fillStyle = COLORS.muted;
      ctx.fillText('SOURCE', 12, 22);
      ctx.fillText('REFERENCE', srcW + 12, 22);
    };

    const drawMatches = (progress) => {
      const sweepX = canvas.width * progress;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.45;
      ctx.strokeStyle = lineColor;

      points.forEach((kp) => {
        const x1 = kp.src.x * scale.srcX;
        const y1 = kp.src.y * scale.srcY;
        const x2 = srcW + kp.ref.x * scale.refX;
        const y2 = kp.ref.y * scale.refY;

        if (x1 > sweepX) return;

        // Clip the connecting line at the sweep front.
        const t = x2 > x1 ? Math.min(1, (sweepX - x1) / (x2 - x1)) : 1;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
        ctx.stroke();

        ctx.globalAlpha = 0.95;
        ctx.fillStyle = pointColor;
        ctx.fillRect(x1 - 1.5, y1 - 1.5, 3, 3);
        if (t >= 1) ctx.fillRect(x2 - 1.5, y2 - 1.5, 3, 3);
        ctx.globalAlpha = 0.45;
      });

      ctx.globalAlpha = 1;
    };

    if (animateSweep && points.length) {
      const startedAt = performance.now();
      const step = (now) => {
        const progress = Math.min(1, (now - startedAt) / SWEEP_DURATION_MS);
        drawFrames();
        drawMatches(progress);
        if (progress < 1) {
          frameRef.current = requestAnimationFrame(step);
        } else {
          completeRef.current?.();
        }
      };
      frameRef.current = requestAnimationFrame(step);
    } else {
      drawFrames();
      if (points.length) drawMatches(1);
    }

    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [sourceImg, referenceImg, keypoints, isFallback, animateSweep]);

  if (!sourceImg || !referenceImg) {
    return <div className="viewport-empty">Load a source and reference frame</div>;
  }

  return <canvas ref={canvasRef} className="viewport-image" />;
}
