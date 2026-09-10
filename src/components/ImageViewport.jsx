import React, { useState } from 'react';
import MatchOverlayCanvas from './MatchOverlayCanvas';
import CompareCanvas from './CompareCanvas';
import AlignedCanvas from './AlignedCanvas';
import ResidualChart from './ResidualChart';
import CoverageGrid from './CoverageGrid';
import { useImage } from '../useImage';
import { TABS } from '../constants';

export default function ImageViewport({
  viewMode,
  setViewMode,
  sourceUrl,
  referenceUrl,
  warpedUrl,
  result,
  status,
  stages,
  animateSweep,
  onSweepComplete
}) {
  const [blendMode, setBlendMode] = useState('split');
  const [splitPosition, setSplitPosition] = useState(50);
  const [gain, setGain] = useState(8);
  const [normalizeCompare, setNormalizeCompare] = useState(true);
  const [showUnregistered, setShowUnregistered] = useState(false);

  const sourceImg = useImage(sourceUrl);
  const referenceImg = useImage(referenceUrl);
  const warpedImg = useImage(warpedUrl);

  const isDone = status === 'COMPLETED';
  const hasResult = isDone && Boolean(warpedImg);
  const needsResult = ['aligned', 'compare', 'residuals', 'coverage'].includes(viewMode);

  const sizeOf = (img) => (img ? `${img.naturalWidth} x ${img.naturalHeight}` : '—');

  const emptyMessage =
    status === 'FAILED'
      ? 'Registration failed — nothing to show here'
      : 'Run registration to populate this view';

  return (
    <section className="left-viewport-panel">
      <div className="viewport-tab-bar">
        {TABS.map((tab, idx) => (
          <button
            key={tab.id}
            className={`viewport-tab-btn ${viewMode === tab.id ? 'active' : ''}`}
            onClick={() => setViewMode(tab.id)}
            title={tab.hint}
          >
            <span className="tab-index mono">{String(idx + 1).padStart(2, '0')}</span>
            <span>{tab.label}</span>
          </button>
        ))}

        <div className="blend-controls">
          {viewMode === 'aligned' && hasResult && (
            <>
              {['split', 'difference', 'checkerboard'].map((mode) => (
                <button
                  key={mode}
                  className={`chip-btn ${blendMode === mode ? 'active' : ''}`}
                  onClick={() => setBlendMode(mode)}
                >
                  {mode}
                </button>
              ))}
              {blendMode === 'split' && (
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitPosition}
                  onChange={(e) => setSplitPosition(Number(e.target.value))}
                  className="split-slider"
                  aria-label="Split position"
                />
              )}
              <button
                className={`chip-btn accent ${showUnregistered ? 'warn' : ''}`}
                onClick={() => setShowUnregistered((v) => !v)}
                title="Swap the warp for the raw source, so you can see the seam break"
              >
                {showUnregistered ? 'showing unregistered' : 'compare to unregistered'}
              </button>
            </>
          )}

          {viewMode === 'compare' && hasResult && (
            <>
              <button
                className={`chip-btn ${normalizeCompare ? 'active' : ''}`}
                onClick={() => setNormalizeCompare((v) => !v)}
                title="Remove illumination differences before comparing, so this measures geometry"
              >
                {normalizeCompare ? 'Normalised' : 'Raw'}
              </button>
              <label className="inline-control">
                <span className="mono">Gain ×{gain}</span>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={gain}
                  onChange={(e) => setGain(Number(e.target.value))}
                  className="split-slider"
                  aria-label="Difference gain"
                />
              </label>
            </>
          )}
        </div>
      </div>

      <div className="viewport-screen">
        <div className="reticle-corner reticle-tl" />
        <div className="reticle-corner reticle-tr" />
        <div className="reticle-corner reticle-bl" />
        <div className="reticle-corner reticle-br" />

        {viewMode === 'preview' && (
          <div className="preview-split">
            <figure>
              {sourceUrl ? (
                <img src={sourceUrl} alt="Source frame" />
              ) : (
                <div className="viewport-empty">No source frame</div>
              )}
              <figcaption className="mono">Source · {sizeOf(sourceImg)}</figcaption>
            </figure>
            <figure>
              {referenceUrl ? (
                <img src={referenceUrl} alt="Reference frame" />
              ) : (
                <div className="viewport-empty">No reference frame</div>
              )}
              <figcaption className="mono">Reference · {sizeOf(referenceImg)}</figcaption>
            </figure>
          </div>
        )}

        {viewMode === 'matches' && (
          <MatchOverlayCanvas
            sourceImg={sourceImg}
            referenceImg={referenceImg}
            keypoints={result?.keypoints}
            isFallback={result?.isFallback}
            animateSweep={animateSweep}
            onSweepComplete={onSweepComplete}
          />
        )}

        {viewMode === 'aligned' &&
          (hasResult ? (
            <AlignedCanvas
              referenceImg={referenceImg}
              warpedImg={warpedImg}
              sourceImg={sourceImg}
              blendMode={blendMode}
              splitPosition={splitPosition}
              showUnregistered={showUnregistered}
            />
          ) : (
            <div className="viewport-empty">{emptyMessage}</div>
          ))}

        {viewMode === 'compare' &&
          (hasResult ? (
            <CompareCanvas
              sourceImg={sourceImg}
              referenceImg={referenceImg}
              warpedImg={warpedImg}
              gain={gain}
              normalize={normalizeCompare}
            />
          ) : (
            <div className="viewport-empty">{emptyMessage}</div>
          ))}

        {viewMode === 'residuals' &&
          (result?.residuals ? (
            <ResidualChart residuals={result.residuals} report={result.report} />
          ) : (
            <div className="viewport-empty">{emptyMessage}</div>
          ))}

        {viewMode === 'coverage' &&
          (result?.uniformity ? (
            <CoverageGrid uniformity={result.uniformity} />
          ) : (
            <div className="viewport-empty">{emptyMessage}</div>
          ))}

        {viewMode !== 'preview' && viewMode !== 'residuals' && viewMode !== 'coverage' && (
          <div className="viewport-coords-tag">
            {viewMode === 'matches' && result?.keypoints?.length > 0 && (
              <span>
                {result.inlierCount.toLocaleString()} inliers
                {result.keypointsSampled && ` · ${result.keypoints.length} drawn`}
              </span>
            )}
            {viewMode === 'aligned' && hasResult && (
              <span>{showUnregistered ? `${blendMode} · unregistered` : blendMode}</span>
            )}
            {viewMode === 'compare' && hasResult && <span>amplified ×{gain}</span>}
          </div>
        )}
      </div>

      <div className="viewport-scrubber-bar">
        <span className="mono label-instrument">Pipeline</span>
        <div className="stage-pips">
          {stages.map((stage) => (
            <span
              key={stage.name}
              className={`stage-pip ${isDone ? 'done' : status === 'RUNNING' ? 'busy' : ''}`}
              title={stage.detail}
            >
              {stage.name}
            </span>
          ))}
        </div>
        {needsResult && !hasResult && status !== 'RUNNING' && (
          <span className="mono val-dim scrubber-note">awaiting run</span>
        )}
      </div>
    </section>
  );
}
