import React, { useCallback, useEffect, useRef, useState } from 'react';
import HeaderBar from './components/HeaderBar';
import ImageViewport from './components/ImageViewport';
import InstrumentPanel from './components/InstrumentPanel';
import BottomBar from './components/BottomBar';
import { DEFAULT_OPTIONS, STAGES } from './constants';

const CUSTOM_PAIR = {
  id: 'custom',
  name: 'Custom pair (upload)',
  summary: 'Register two images of your own.',
  condition: 'same_sensor',
  gsdRatio: '1:1',
  synthetic: false
};

export default function App() {
  const [pairs, setPairs] = useState([]);
  const [matchers, setMatchers] = useState([]);
  const [selectedPairId, setSelectedPairId] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const [viewMode, setViewMode] = useState('preview');
  const [status, setStatus] = useState('IDLE'); // IDLE | RUNNING | COMPLETED | FAILED
  const [result, setResult] = useState(null);
  const [failureReason, setFailureReason] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);

  const [customFiles, setCustomFiles] = useState({ source: null, reference: null });
  const [customPreviews, setCustomPreviews] = useState({ source: null, reference: null });

  const [animateSweep, setAnimateSweep] = useState(false);
  const elapsedTimerRef = useRef(null);

  const activePair =
    selectedPairId === 'custom'
      ? CUSTOM_PAIR
      : pairs.find((p) => p.id === selectedPairId) || null;

  // The manifest on the server is the single source of truth for which pairs
  // exist, and which matchers this install can actually run.
  useEffect(() => {
    let cancelled = false;

    fetch('/api/samples')
      .then((res) => {
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setPairs(data.pairs);
        setMatchers(data.matchers || []);
        setSelectedPairId(data.pairs.length ? data.pairs[0].id : 'custom');
        if (!data.pairs.length) {
          setLoadError('No dataset pairs found. Run: python prepare_datasets.py');
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setLoadError(`Cannot reach the backend (${err.message}). Start it with: python main.py`);
        setSelectedPairId('custom');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const resetRun = useCallback(() => {
    setStatus('IDLE');
    setResult(null);
    setFailureReason(null);
    setElapsedMs(0);
    setAnimateSweep(false);
  }, []);

  useEffect(resetRun, [selectedPairId, resetRun]);

  useEffect(() => () => clearInterval(elapsedTimerRef.current), []);

  const handleCustomUpload = useCallback(
    (slot, file) => {
      setCustomFiles((prev) => ({ ...prev, [slot]: file }));
      setCustomPreviews((prev) => {
        if (prev[slot]) URL.revokeObjectURL(prev[slot]);
        return { ...prev, [slot]: URL.createObjectURL(file) };
      });
      resetRun();
    },
    [resetRun]
  );

  const setOption = useCallback((key, value) => {
    setOptions((prev) => ({ ...prev, [key]: value }));
  }, []);

  const runRegistration = async () => {
    if (status === 'RUNNING' || !activePair) return;

    const isCustom = selectedPairId === 'custom';
    if (isCustom && !(customFiles.source && customFiles.reference)) {
      setStatus('FAILED');
      setFailureReason('Load both a source and a reference frame before running.');
      return;
    }

    setStatus('RUNNING');
    setResult(null);
    setFailureReason(null);
    setAnimateSweep(false);

    // The backend cannot report progress mid-run, so show real elapsed time
    // rather than simulating stage-by-stage advancement.
    const startedAt = performance.now();
    setElapsedMs(0);
    elapsedTimerRef.current = setInterval(
      () => setElapsedMs(Math.round(performance.now() - startedAt)),
      100
    );

    try {
      const formData = new FormData();
      if (isCustom) {
        formData.append('src_file', customFiles.source);
        formData.append('ref_file', customFiles.reference);
        formData.append('condition', 'same_sensor');
      } else {
        formData.append('pair_id', activePair.id);
      }
      Object.entries(options).forEach(([key, value]) => formData.append(key, value));

      const response = await fetch('/api/register', { method: 'POST', body: formData });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `server returned ${response.status}`);
      }

      const data = await response.json();

      if (data.status === 'failed') {
        setStatus('FAILED');
        setFailureReason(data.failureReason);
        return;
      }

      setResult(data);
      setStatus('COMPLETED');
      setAnimateSweep(true);
      setViewMode((mode) => (mode === 'preview' ? 'matches' : mode));
    } catch (err) {
      setStatus('FAILED');
      setFailureReason(err.message);
    } finally {
      clearInterval(elapsedTimerRef.current);
      setElapsedMs(Math.round(performance.now() - startedAt));
    }
  };

  const sourceUrl = selectedPairId === 'custom' ? customPreviews.source : activePair?.sourceUrl;
  const referenceUrl =
    selectedPairId === 'custom' ? customPreviews.reference : activePair?.referenceUrl;

  return (
    <div className="app-container">
      <div className="starfield" aria-hidden="true" />

      <HeaderBar activePair={activePair} status={status} result={result} />

      <main className="main-viewport-grid">
        <ImageViewport
          viewMode={viewMode}
          setViewMode={setViewMode}
          sourceUrl={sourceUrl}
          referenceUrl={referenceUrl}
          warpedUrl={result?.warpedDataUrl}
          result={result}
          status={status}
          stages={STAGES}
          animateSweep={animateSweep}
          onSweepComplete={() => setAnimateSweep(false)}
        />

        <InstrumentPanel
          activePair={activePair}
          status={status}
          result={result}
          failureReason={failureReason}
          elapsedMs={elapsedMs}
          stages={STAGES}
        />
      </main>

      <BottomBar
        pairs={pairs}
        matchers={matchers}
        customPair={CUSTOM_PAIR}
        selectedPairId={selectedPairId}
        onSelectPair={setSelectedPairId}
        options={options}
        onOptionChange={setOption}
        onRun={runRegistration}
        isRunning={status === 'RUNNING'}
        onCustomUpload={handleCustomUpload}
        customFiles={customFiles}
        loadError={loadError}
      />
    </div>
  );
}
