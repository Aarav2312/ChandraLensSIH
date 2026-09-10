import React, { useEffect, useRef } from 'react';

/**
 * Composites the warp the backend produced against the reference.
 *
 * A good registration makes every one of these modes look like a single
 * uninterrupted image, which is the right answer but reads as "nothing
 * happened". The `showUnregistered` toggle swaps the warp for the raw source
 * so the seam visibly breaks — the alignment is only legible next to its
 * absence.
 */
function label(ctx, text, x, y, color, scale) {
  const size = Math.max(11, Math.round(13 * scale));
  ctx.font = `${size}px "IBM Plex Mono", monospace`;
  const padding = size * 0.45;
  const width = ctx.measureText(text).width;

  ctx.fillStyle = 'rgba(6, 9, 14, 0.82)';
  ctx.fillRect(x - padding, y - size, width + padding * 2, size * 1.6);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y + size * 0.25);
}

export default function AlignedCanvas({
  referenceImg,
  warpedImg,
  sourceImg,
  blendMode,
  splitPosition,
  showUnregistered
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !referenceImg || !warpedImg) return;

    const overlay = showUnregistered ? sourceImg || warpedImg : warpedImg;

    canvas.width = referenceImg.naturalWidth;
    canvas.height = referenceImg.naturalHeight;
    const { width, height } = canvas;
    const ctx = canvas.getContext('2d');
    const scale = width / 640;

    const overlayName = showUnregistered ? 'SOURCE, UNREGISTERED' : 'WARPED SOURCE';
    const overlayColor = showUnregistered ? '#c9a25a' : '#7fb2c4';

    ctx.clearRect(0, 0, width, height);

    if (blendMode === 'split') {
      const splitX = Math.round((width * splitPosition) / 100);

      ctx.drawImage(referenceImg, 0, 0, width, height);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, height);
      ctx.clip();
      ctx.drawImage(overlay, 0, 0, width, height);
      ctx.restore();

      ctx.strokeStyle = overlayColor;
      ctx.lineWidth = Math.max(1, scale);
      ctx.beginPath();
      ctx.moveTo(splitX + 0.5, 0);
      ctx.lineTo(splitX + 0.5, height);
      ctx.stroke();

      if (splitX > width * 0.22) {
        label(ctx, overlayName, 14 * scale, 26 * scale, overlayColor, scale);
      }
      if (splitX < width * 0.78) {
        label(ctx, 'REFERENCE', splitX + 14 * scale, 26 * scale, '#8990a0', scale);
      }
    } else if (blendMode === 'difference') {
      ctx.drawImage(overlay, 0, 0, width, height);
      ctx.globalCompositeOperation = 'difference';
      ctx.drawImage(referenceImg, 0, 0, width, height);
      ctx.globalCompositeOperation = 'source-over';
      label(ctx, `${overlayName} − REFERENCE`, 14 * scale, 26 * scale, overlayColor, scale);
    } else {
      const cell = Math.max(24, Math.round(width / 12));
      ctx.drawImage(referenceImg, 0, 0, width, height);
      for (let y = 0; y < height; y += cell) {
        for (let x = 0; x < width; x += cell) {
          if (((x / cell) | 0) % 2 === ((y / cell) | 0) % 2) continue;
          ctx.save();
          ctx.beginPath();
          ctx.rect(x, y, cell, cell);
          ctx.clip();
          ctx.drawImage(overlay, 0, 0, width, height);
          ctx.restore();
        }
      }
      label(ctx, `${overlayName} / REFERENCE TILES`, 14 * scale, 26 * scale, overlayColor, scale);
    }
  }, [referenceImg, warpedImg, sourceImg, blendMode, splitPosition, showUnregistered]);

  return <canvas ref={canvasRef} className="viewport-image" />;
}
