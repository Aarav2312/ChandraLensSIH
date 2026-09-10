import React from 'react';

/**
 * Distribution of per-correspondence reprojection error. A tight distribution
 * hugging zero means the homography explains every match; a long tail means
 * some inliers only just survived MAGSAC++.
 */
export default function ResidualChart({ residuals, report }) {
  const { histogram, bin_edges: edges } = residuals;
  const peak = Math.max(...histogram, 1);
  const limit = report?.rmse_limit_px;

  // Where the condition's accuracy budget falls along the x axis.
  const axisMax = edges[edges.length - 1] || 1;
  const budgetPct = limit != null ? Math.min(100, (limit / axisMax) * 100) : null;

  const stats = [
    ['RMSE', residuals.rmse_px],
    ['Median', residuals.median_px],
    ['Mean', residuals.mean_px],
    ['95th pct', residuals.p95_px],
    ['Max', residuals.max_px]
  ];

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <span className="card-title">Reprojection error per correspondence</span>
        <span className="mono val-dim">{axisMax.toFixed(2)} px full scale</span>
      </div>

      <div className="histogram" role="img" aria-label="Histogram of reprojection residuals">
        {budgetPct != null && budgetPct < 100 && (
          <div className="histogram-budget" style={{ left: `${budgetPct}%` }}>
            <span className="mono">{limit} px budget</span>
          </div>
        )}
        {histogram.map((count, idx) => (
          <div
            key={idx}
            className="histogram-bar"
            style={{ height: `${(count / peak) * 100}%` }}
            title={`${edges[idx].toFixed(2)}–${edges[idx + 1].toFixed(2)} px: ${count}`}
          />
        ))}
      </div>

      <div className="histogram-axis mono">
        <span>0</span>
        <span>{(axisMax / 2).toFixed(2)} px</span>
        <span>{axisMax.toFixed(2)} px</span>
      </div>

      <div className="chart-stats">
        {stats.map(([label, value]) => (
          <div key={label} className="chart-stat">
            <span className="chart-stat-label">{label}</span>
            <span className={`chart-stat-value mono ${value < 1 ? 'val-cyan' : 'val-amber'}`}>
              {value.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
