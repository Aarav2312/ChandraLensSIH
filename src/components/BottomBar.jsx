import React, { useRef, useState } from 'react';

export default function BottomBar({
  pairs,
  matchers,
  customPair,
  selectedPairId,
  onSelectPair,
  options,
  onOptionChange,
  onRun,
  isRunning,
  onCustomUpload,
  customFiles,
  loadError
}) {
  const sourceInputRef = useRef(null);
  const refInputRef = useRef(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const isCustom = selectedPairId === 'custom';
  const activePair = isCustom ? customPair : pairs.find((p) => p.id === selectedPairId);
  const canRun = isCustom
    ? Boolean(customFiles.source && customFiles.reference)
    : Boolean(activePair);
  const isCrossModal = activePair?.condition === 'cross_modal';

  const handleFile = (slot) => (event) => {
    const file = event.target.files?.[0];
    if (file) onCustomUpload(slot, file);
  };

  return (
    <footer className="bottom-control-bar">
      <div className="bottom-left-controls">
        <select
          className="instrument-select"
          value={selectedPairId || ''}
          onChange={(e) => onSelectPair(e.target.value)}
          disabled={isRunning}
          aria-label="Dataset pair"
        >
          {pairs.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} — {p.gsdRatio} GSD
            </option>
          ))}
          <option value="custom">{customPair.name}</option>
        </select>

        <select
          className="instrument-select narrow"
          value={options.matcher}
          onChange={(e) => onOptionChange('matcher', e.target.value)}
          disabled={isRunning}
          aria-label="Matcher"
        >
          {matchers.map((m) => (
            <option key={m.id} value={m.id} title={m.detail}>
              {m.name}
              {m.family !== 'Hybrid' ? ` · ${m.family}` : ''}
            </option>
          ))}
        </select>

        {isCustom && (
          <div className="upload-group">
            <input type="file" ref={sourceInputRef} accept="image/*" hidden onChange={handleFile('source')} />
            <input type="file" ref={refInputRef} accept="image/*" hidden onChange={handleFile('reference')} />
            <button
              className={`upload-trigger-btn ${customFiles.source ? 'loaded' : ''}`}
              onClick={() => sourceInputRef.current?.click()}
              disabled={isRunning}
            >
              {customFiles.source ? `Src: ${customFiles.source.name}` : 'Load source'}
            </button>
            <button
              className={`upload-trigger-btn ${customFiles.reference ? 'loaded' : ''}`}
              onClick={() => refInputRef.current?.click()}
              disabled={isRunning}
            >
              {customFiles.reference ? `Ref: ${customFiles.reference.name}` : 'Load reference'}
            </button>
          </div>
        )}

        <button
          className={`chip-btn ${showAdvanced ? 'active' : ''}`}
          onClick={() => setShowAdvanced((v) => !v)}
          aria-expanded={showAdvanced}
        >
          Tuning
        </button>

        {showAdvanced && (
          <div className="tuning-popover">
            <label className="tuning-row">
              <span>RANSAC threshold</span>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={options.ransac_threshold}
                onChange={(e) => onOptionChange('ransac_threshold', Number(e.target.value))}
                disabled={isRunning}
              />
              <span className="mono val-cyan">{options.ransac_threshold.toFixed(1)} px</span>
            </label>

            <label className="tuning-row">
              <span>Gamma</span>
              <input
                type="range"
                min="0.4"
                max="2.0"
                step="0.05"
                value={options.gamma}
                onChange={(e) => onOptionChange('gamma', Number(e.target.value))}
                disabled={isRunning}
              />
              <span className="mono val-cyan">{options.gamma.toFixed(2)}</span>
            </label>

            <label className="tuning-row">
              <span>Denoise</span>
              <input
                type="range"
                min="0"
                max="3"
                step="0.25"
                value={options.denoise}
                onChange={(e) => onOptionChange('denoise', Number(e.target.value))}
                disabled={isRunning}
              />
              <span className="mono val-cyan">
                {options.denoise === 0 ? 'off' : options.denoise.toFixed(2)}
              </span>
            </label>

            <label className="tuning-row checkbox">
              <input
                type="checkbox"
                checked={options.use_clahe}
                onChange={(e) => onOptionChange('use_clahe', e.target.checked)}
                disabled={isRunning}
              />
              <span>CLAHE contrast equalisation</span>
            </label>

            <div className="tuning-divider">Cross-modal</div>

            <label className="tuning-row">
              <span>Representation</span>
              <select
                className="instrument-select"
                value={options.representation}
                onChange={(e) => onOptionChange('representation', e.target.value)}
                disabled={isRunning}
              >
                <option value="intensity">Intensity</option>
                <option value="structural">Phase congruency</option>
              </select>
              <span />
            </label>

            <label className="tuning-row">
              <span>Refinement</span>
              <select
                className="instrument-select"
                value={options.similarity}
                onChange={(e) => onOptionChange('similarity', e.target.value)}
                disabled={isRunning}
              >
                <option value="ncc">NCC</option>
                <option value="mi">Mutual information</option>
              </select>
              <span />
            </label>

            {isCrossModal && options.representation === 'intensity' && (
              <p className="tuning-note warn">
                This pair has no shared intensity relationship. Intensity matching and NCC will
                both degrade — switch to phase congruency and mutual information.
              </p>
            )}

            <p className="tuning-note">
              Changing any of these re-runs the pipeline for real; results are not interpolated.
            </p>
          </div>
        )}

        {!isCustom && activePair && <span className="pair-summary">{activePair.summary}</span>}
        {loadError && <span className="pair-summary error">{loadError}</span>}
      </div>

      <button
        className={`action-run-btn ${isRunning ? 'running' : ''}`}
        onClick={onRun}
        disabled={isRunning || !canRun}
      >
        {isRunning ? (
          <>
            <span>Running</span>
            <span className="blinking-cursor" />
          </>
        ) : (
          'Run registration'
        )}
      </button>
    </footer>
  );
}
