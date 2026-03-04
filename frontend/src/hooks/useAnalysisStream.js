/**
 * C.R.U.Y.F.F. — useAnalysisStream (Revised)
 *
 * Fixes applied:
 * - Fix F6: TelemetryRingBuffer for video-sync AR mode
 * - Fix F9: Critical alerts bypass 4Hz throttle
 *
 * Exposes:
 * - dataRef: mutable ref for useFrame (latest frame in live mode)
 * - ringBuffer: TelemetryRingBuffer instance for video-sync queries
 * - hudState: React state for HUD (4Hz + critical bypass)
 * - status: connection status
 * - requestGhost: multiplexed Ghost RPC (Fix F5)
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createAnalysisStream } from '../lib/transport';
import { TelemetryRingBuffer } from '../lib/telemetryBuffer';

export function useAnalysisStream() {
    // Mutable ref — Three.js reads at 60-120fps (Fix F1)
    const dataRef = useRef({
        voids: [],
        bio_alerts: [],
        ghost: null,
        timestamp: 0,
        coordinates: null,
    });

    // Telemetry ring buffer for AR video sync (Fix F6)
    const ringBufferRef = useRef(new TelemetryRingBuffer(750));

    // React state — 4Hz + critical bypass (Fix F9)
    const [hudState, setHudState] = useState({
        voidCount: 0,
        maxPersistence: 0,
        bioAlerts: [],
        ghostXt: null,
    });

    const [status, setStatus] = useState('disconnected');
    const streamRef = useRef(null);

    const buildHudState = useCallback((data) => ({
        voidCount: (data.voids || []).length,
        maxPersistence: (data.voids || []).length > 0
            ? Math.max(...(data.voids || []).map((v) => v.persistence))
            : 0,
        bioAlerts: (data.bio_alerts || []).filter(
            (a) => a.status === 'amber' || a.status === 'red'
        ),
        ghostXt: data.ghost?.expected_xt ?? null,
    }), []);

    // Standard 4Hz update
    const onHudUpdate = useCallback((data) => {
        setHudState(buildHudState(data));
    }, [buildHudState]);

    // Fix F9: Critical bypass — immediate update
    const onCriticalAlert = useCallback((data) => {
        setHudState(buildHudState(data));
    }, [buildHudState]);

    useEffect(() => {
        const stream = createAnalysisStream({
            dataRef,
            ringBuffer: ringBufferRef.current,
            onHudUpdate,
            onCriticalAlert,
            onStatusChange: setStatus,
        });
        streamRef.current = stream;
        return () => stream.close();
    }, [onHudUpdate, onCriticalAlert]);

    // Multiplexed Ghost RPC (Fix F5)
    const requestGhost = useCallback((playerId) => {
        streamRef.current?.send({
            action: 'request_ghost',
            player_id: playerId,
        });
    }, []);

    return {
        dataRef,
        ringBuffer: ringBufferRef.current,
        hudState,
        status,
        requestGhost,
    };
}
