import React from 'react';

/**
 * Spatial distribution of verified inliers over the source frame.
 *
 * The problem statement asks for correspondences with a uniform distribution
 * across the image, not merely a low error: a homography fitted from matches
 * clustered in one region extrapolates poorly everywhere else. This is that
 * requirement made visible and scored.
 */
export default function CoverageGrid({ uniformity }) {
  const { grid, occupancy, occupied_cells: occupied, coverage, score } = uniformity;
  const peak = Math.max(...occupancy, 1);

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <span className="card-title">Inlier distribution across the source frame</span>
        <span className="mono val-dim">
          {grid}×{grid} cells
        </span>
      </div>

      <div
        className="coverage-grid"
        style={{ gridTemplateColumns: `repeat(${grid}, 1fr)` }}
        role="img"
        aria-label={`Coverage heat map, ${occupied} of ${grid * grid} cells occupied`}
      >
        {occupancy.map((count, idx) => (
          <div
            key={idx}
            className={`coverage-cell ${count === 0 ? 'empty' : ''}`}
            style={{ opacity: count === 0 ? 1 : 0.25 + (count / peak) * 0.75 }}
            title={`${count} inlier${count === 1 ? '' : 's'}`}
          >
            <span className="mono">{count || ''}</span>
          </div>
        ))}
      </div>

      <div className="chart-stats">
        <div className="chart-stat">
          <span className="chart-stat-label">Uniformity score</span>
          <span className={`chart-stat-value mono ${score > 0.5 ? 'val-cyan' : 'val-amber'}`}>
            {score.toFixed(3)}
          </span>
        </div>
        <div className="chart-stat">
          <span className="chart-stat-label">Cells occupied</span>
          <span className="chart-stat-value mono">
            {occupied} / {grid * grid}
          </span>
        </div>
        <div className="chart-stat">
          <span className="chart-stat-label">Coverage</span>
          <span className="chart-stat-value mono">{(coverage * 100).toFixed(1)}%</span>
        </div>
      </div>

      <p className="chart-note">
        Empty cells are regions the matcher found no verified correspondence in. Score combines
        coverage with how evenly the occupied cells are populated.
      </p>
    </div>
  );
}
