// The six pipeline stages, in execution order. Names match the keys the backend
// returns in `timings`; durations are never hardcoded here, they come from the
// server after a run.
export const STAGES = [
  { name: 'NORMALIZE', detail: 'CLAHE equalisation, GSD resampling' },
  { name: 'MATCH', detail: 'Dense or classical correspondence' },
  { name: 'VERIFY', detail: 'MAGSAC++ outlier rejection' },
  { name: 'ALIGN', detail: 'Projective warp to reference frame' },
  { name: 'REFINE', detail: 'NCC sub-pixel refinement' },
  { name: 'REPORT', detail: 'Condition-stratified evaluation' }
];

export const TABS = [
  { id: 'preview', label: 'Preview', hint: 'Source and reference side by side' },
  { id: 'matches', label: 'Matches', hint: 'Verified correspondences' },
  { id: 'aligned', label: 'Aligned', hint: 'Warped product over the reference' },
  { id: 'compare', label: 'Compare', hint: 'Residual before and after registration' },
  { id: 'residuals', label: 'Residuals', hint: 'Reprojection error distribution' },
  { id: 'coverage', label: 'Coverage', hint: 'Spatial distribution of inliers' }
];

export const COLORS = {
  primary: '#7FA8B3',
  primaryBright: '#9DC2CC',
  fallback: '#C9A25A',
  fallbackBright: '#E0B86C',
  muted: '#8A8D91',
  hairline: '#2A2C2F'
};

export const DEFAULT_OPTIONS = {
  matcher: 'auto',
  ransac_threshold: 3.0,
  use_clahe: true,
  gamma: 1.0,
  denoise: 0.0,
  representation: 'intensity',
  similarity: 'ncc'
};
