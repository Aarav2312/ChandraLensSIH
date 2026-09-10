import React from 'react';

export default function HeaderBar({ activePair, status, result }) {
  const isFallback = Boolean(result?.isFallback);

  const statusText =
    status === 'RUNNING'
      ? 'Running'
      : status === 'FAILED'
      ? 'Failed'
      : status === 'COMPLETED'
      ? isFallback
        ? 'Complete (fallback)'
        : 'Complete'
      : 'Ready';

  const isAlert = status === 'FAILED' || (isFallback && status === 'COMPLETED');

  return (
    <header className="header-bar">
      <div className="header-left">
        <div className="header-title-main">
          <span>Chandra Lens</span>
          <span className="header-sep">/</span>
          <span className="header-subtitle">Lunar image registration</span>
        </div>
        <div className="header-badge-tag">LoFTR · MAGSAC++ · NCC sub-pixel</div>
      </div>

      <div className="header-right">
        {activePair && (
          <div className="telemetry-node">
            <span className="mono label-instrument">Pair</span>
            <span className="val mono">{activePair.name}</span>
          </div>
        )}

        <div className="telemetry-node">
          <div className={`status-dot ${isAlert ? 'amber' : ''} ${status === 'RUNNING' ? 'pulsing' : ''}`} />
          <span className={`mono ${isAlert ? 'val-amber' : 'val-cyan'}`}>{statusText}</span>
        </div>
      </div>
    </header>
  );
}
