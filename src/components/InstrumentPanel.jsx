import React from 'react';

function Row({ label, children, divider }) {
  return (
    <div className={`readout-row${divider ? ' divider' : ''}`}>
      <span className="readout-key">{label}</span>
      <span className="readout-val mono">{children}</span>
    </div>
  );
}

const PENDING = <span className="val-dim">—</span>;

function formatMs(ms) {
  if (ms == null) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export default function InstrumentPanel({
  activePair,
  status,
  result,
  failureReason,
  elapsedMs,
  stages
}) {
  const isRunning = status === 'RUNNING';
  const isFailed = status === 'FAILED';
  const isDone = status === 'COMPLETED';

  const report = result?.report;
  const groundTruth = report?.ground_truth;
  const isFallback = Boolean(result?.isFallback);
  const truth = groundTruth?.transform;

  const statusLabel = isRunning
    ? 'RUNNING'
    : isFailed
    ? 'FAILED'
    : isDone
    ? isFallback
      ? 'COMPLETE (FALLBACK)'
      : 'COMPLETE'
    : 'IDLE';

  const statusClass = isFailed
    ? 'val-amber'
    : isFallback && isDone
    ? 'val-amber'
    : isDone || isRunning
    ? 'val-cyan'
    : 'val-dim';

  return (
    <aside className="right-instrument-panel">
      <div className="instrument-panel-header">
        <span>Run report</span>
        <span className="mono val-dim">{activePair?.gsdRatio || '—'} GSD</span>
      </div>

      <div className="instrument-scroll-body">
        <div className={`readout-card${isRunning ? ' active-stage' : ''}`}>
          <Row label="Status">
            <span className={statusClass}>
              {statusLabel}
              {isRunning && <span className="blinking-cursor" />}
            </span>
          </Row>
          <Row label="Matcher">
            {result ? (
              <span className={isFallback ? 'val-amber' : 'val-cyan'}>{result.method}</span>
            ) : (
              PENDING
            )}
          </Row>
          <Row label="Elapsed" divider>
            {isRunning || isDone || isFailed ? formatMs(result?.processingTimeMs ?? elapsedMs) : PENDING}
          </Row>

          {/* Every number in this panel depends on these, and they persist
              across pair changes — so show what actually produced this run
              rather than leaving a tweaked slider to silently explain a
              difference later. */}
          {result?.options && (
            <div className="settings-line mono">
              {[
                result.options.representation !== 'intensity' && result.options.representation,
                result.options.similarity !== 'ncc' && result.options.similarity.toUpperCase(),
                `RANSAC ${result.options.ransac_threshold}px`,
                !result.options.use_clahe && 'no CLAHE',
                result.options.gamma !== 1 && `γ ${result.options.gamma}`,
                result.options.denoise > 0 && `denoise ${result.options.denoise}`
              ]
                .filter(Boolean)
                .join(' · ')}
            </div>
          )}
        </div>

        {isFailed && failureReason && (
          <div className="failure-alert-box">
            <div className="failure-alert-title">Registration failed</div>
            <div className="failure-alert-body mono">{failureReason}</div>
          </div>
        )}

        {isFallback && isDone && (
          <div className="notice-box amber">
            <div className="notice-title">Fallback matcher used</div>
            <div className="notice-body">
              LoFTR was unavailable or returned too few confident correspondences, so the
              classical detector produced this result.
            </div>
          </div>
        )}

        <div className="readout-card">
          <Row label="Correspondences">
            {result ? <span className="val-cyan">{result.matchCount.toLocaleString()}</span> : PENDING}
          </Row>
          <Row label="Mean confidence">
            {result ? `${result.confidence.toFixed(1)}%` : PENDING}
          </Row>
          <Row label="Inliers (MAGSAC++)">
            {result ? (
              <span className={isFallback ? 'val-amber' : 'val-cyan'}>
                {result.inlierCount.toLocaleString()} · {result.inlierRatio.toFixed(1)}%
              </span>
            ) : (
              PENDING
            )}
          </Row>
          <Row label="Residual RMSE" divider>
            {result ? (
              <span className={result.residualRmse < 1.0 ? 'val-cyan' : 'val-amber'}>
                {result.residualRmse.toFixed(3)} px
              </span>
            ) : (
              PENDING
            )}
          </Row>
        </div>

        {/* Accuracy against the known transform. Only shown for pairs that have
            one recorded, because it is the only figure here the estimator
            cannot influence. */}
        {groundTruth && (
          <div className="readout-card highlight">
            <div className="card-title">Measured against ground truth</div>
            <Row label="Mean corner error">
              <span className="val-cyan">{groundTruth.mean_corner_error_px.toFixed(3)} px</span>
            </Row>
            <Row label="Max corner error">
              <span className="val-cyan">{groundTruth.max_corner_error_px.toFixed(3)} px</span>
            </Row>
            <div className="card-note">
              Corner displacement between the estimated homography and the transform used to
              build this pair. Independent of the estimator's own residual.
            </div>
          </div>
        )}

        {/* Decomposed from the solved homography, not reported by the solver.
            The "true" column decomposes the ground-truth matrix with the same
            routine, so the two are directly comparable. */}
        {result?.transform && (
          <div className="readout-card">
            <div className="card-title-row">
              <span className="card-title">Recovered transform</span>
              {truth && <span className="mono val-dim">vs true</span>}
            </div>

            <Row label="Scale">
              <span className="val-cyan">{result.transform.scale.toFixed(5)}</span>
              {truth && <span className="val-dim"> / {truth.scale.toFixed(5)}</span>}
            </Row>
            <Row label="Rotation">
              <span className="val-cyan">{result.transform.rotation_deg.toFixed(4)}°</span>
              {truth && <span className="val-dim"> / {truth.rotation_deg.toFixed(4)}°</span>}
            </Row>
            <Row label="Translation">
              <span className="val-cyan">
                {result.transform.tx_px.toFixed(2)}, {result.transform.ty_px.toFixed(2)} px
              </span>
            </Row>
            <Row label="Shear" divider>
              {result.transform.shear.toExponential(2)}
            </Row>
          </div>
        )}

        {result?.uniformity && (
          <div className="readout-card">
            <div className="card-title-row">
              <span className="card-title">Distribution</span>
              <span
                className={`grade-chip ${result.uniformity.score > 0.5 ? '' : 'amber'}`}
              >
                {result.uniformity.score.toFixed(2)}
              </span>
            </div>
            <Row label="Cells occupied">
              {result.uniformity.occupied_cells} / {result.uniformity.grid ** 2}
            </Row>
            <Row label="Coverage" divider>
              {(result.uniformity.coverage * 100).toFixed(1)}%
            </Row>
          </div>
        )}

        {report && (
          <div className="readout-card">
            <div className="card-title-row">
              <span className="card-title">Evaluation</span>
              <span className={`grade-chip ${report.evaluation_grade === 'OPTIMAL' ? '' : 'amber'}`}>
                {report.evaluation_grade}
              </span>
            </div>

            <div className="condition-label">{report.condition_label}</div>

            <Row label="Residual budget">
              <span className={report.meets_rmse_budget ? 'val-cyan' : 'val-amber'}>
                {report.residual_rmse_px.toFixed(3)} / {report.rmse_limit_px.toFixed(1)} px
              </span>
            </Row>
            <Row label="Inlier target">
              <span className={report.meets_inlier_budget ? 'val-cyan' : 'val-amber'}>
                {report.inlier_ratio_pct.toFixed(1)} / {report.inlier_target_pct.toFixed(0)}%
              </span>
            </Row>
            <Row label="Sub-pixel">
              <span className={report.subpixel ? 'val-cyan' : 'val-amber'}>
                {report.subpixel ? 'Yes' : 'No'}
              </span>
            </Row>
            <Row label="Refined points" divider>
              {report.refined_count.toLocaleString()} of {result.inlierCount.toLocaleString()} by NCC
            </Row>
          </div>
        )}

        <div className="readout-card">
          <div className="card-title-row">
            <span className="card-title">Homography</span>
            <span className="mono val-dim">{result ? 'solved' : 'not solved'}</span>
          </div>

          {result ? (
            <div className="matrix-block">
              {result.homographyMatrix.flat().map((val, idx) => (
                <div key={idx} className="matrix-cell mono">
                  {Math.abs(val) < 0.01 ? val.toExponential(2) : val.toFixed(5)}
                </div>
              ))}
            </div>
          ) : (
            <div className="matrix-placeholder mono val-dim">not solved</div>
          )}
        </div>

        <div className="readout-card">
          <div className="card-title-row">
            <span className="card-title">Stage timings</span>
            <span className="mono val-dim">
              {result ? formatMs(result.processingTimeMs) : '—'}
            </span>
          </div>

          <div className="stage-list">
            {stages.map((stage, idx) => {
              const ms = result?.timings?.[stage.name];
              return (
                <div key={stage.name} className="stage-row">
                  <span className="stage-name">
                    <span className="val-dim">{idx + 1}.</span> {stage.name}
                  </span>
                  <span className="stage-detail">{stage.detail}</span>
                  <span className="stage-time mono">
                    {ms != null ? `${ms.toFixed(1)} ms` : isFailed ? '—' : 'queued'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {activePair?.synthetic && (
          <div className="notice-box">
            <div className="notice-title">Synthetic pair</div>
            <div className="notice-body">
              {activePair.provenance?.note} Imagery credit: {activePair.provenance?.credit}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
