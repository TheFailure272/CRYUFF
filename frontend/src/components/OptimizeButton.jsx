/**
 * C.R.U.Y.F.F. — Optimize Button
 *
 * Triggers Ghost RPC via the multiplexed WebSocket.
 * - 3-second cooldown between presses
 * - Spinner while computing
 * - Touch-optimized (48px min height)
 */

import { useState, useCallback, useRef } from 'react';

const COOLDOWN_MS = 3000;

export default function OptimizeButton({ requestGhost }) {
    const [loading, setLoading] = useState(false);
    const cooldownRef = useRef(false);

    const handlePress = useCallback(() => {
        if (cooldownRef.current || loading) return;

        // Default to player_id=10 (the #10 — classic playmaker)
        requestGhost(10);
        setLoading(true);
        cooldownRef.current = true;

        // Simulate ghost compute time + enforce cooldown
        setTimeout(() => {
            setLoading(false);
        }, 1500);

        setTimeout(() => {
            cooldownRef.current = false;
        }, COOLDOWN_MS);
    }, [requestGhost, loading]);

    return (
        <button
            className="optimize-btn"
            onClick={handlePress}
            disabled={loading || cooldownRef.current}
            id="optimize-ghost-btn"
        >
            {loading ? (
                <>
                    <div className="spinner" />
                    COMPUTING…
                </>
            ) : (
                <>⚡ OPTIMIZE</>
            )}
        </button>
    );
}
