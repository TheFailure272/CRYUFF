/**
 * C.R.U.Y.F.F. — App (Final Deployment-Hardened)
 *
 * Root layout: Video → PitchCanvas → HUD overlay.
 *
 * Fixes wired:
 * - Fix F1:  useRef + useFrame (no React reconciliation)
 * - Fix F5:  Multiplexed WebSocket (single connection)
 * - Fix F6:  TelemetryRingBuffer (for video sync)
 * - Fix F9:  Selective throttle bypass (red alerts)
 * - Fix F13: PTZ Homography (via PitchCanvas)
 * - Fix F14: WebGL context recovery (via PitchCanvas)
 * - Fix F16: Battery degradation (via PitchCanvas → layers)
 */

import { useRef } from 'react';
import PitchCanvas from './components/PitchCanvas';
import HUD from './components/HUD';
import { useAnalysisStream } from './hooks/useAnalysisStream';
import { useDegradation, DegradationLevel } from './lib/degradation';

export default function App() {
    const { dataRef, hudState, status, requestGhost } = useAnalysisStream();
    const degradation = useDegradation();
    const videoRef = useRef(null);

    return (
        <div className="app-container">
            {/* ── Video Layer (uncomment for AR overlay mode) ─────────── */}
            {/* <video
        ref={videoRef}
        className="video-layer"
        src="/broadcast.mp4"
        autoPlay muted loop
      /> */}

            {/* ── Three.js Pitch Canvas ──────────────────────────────── */}
            <PitchCanvas dataRef={dataRef} degradation={degradation} />

            {/* ── HUD Overlay ────────────────────────────────────────── */}
            <HUD
                hudState={hudState}
                status={status}
                requestGhost={requestGhost}
                degradation={degradation}
            />

            {/* ── Battery Warning ────────────────────────────────────── */}
            {degradation.level !== DegradationLevel.FULL && (
                <div
                    className="system-tag"
                    style={{
                        color: degradation.level === DegradationLevel.MINIMAL
                            ? 'var(--pulse-red)' : 'var(--pulse-amber)',
                        bottom: 36,
                    }}
                >
                    ⚡ {degradation.level === DegradationLevel.MINIMAL
                        ? 'LOW BATTERY — MINIMAL MODE'
                        : 'REDUCED POWER MODE'
                    } ({Math.round(degradation.battery * 100)}%)
                </div>
            )}
        </div>
    );
}
