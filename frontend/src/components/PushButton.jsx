/**
 * C.R.U.Y.F.F. — PushButton (Feature 5)
 *
 * "⬆ PUSH" button that sends tactical insights to the
 * dressing room Apple TV for half-time review.
 *
 * Fix F38: All communication is LAN-only (192.168.x.x).
 * The button sends a POST to the edge server, which publishes
 * raw telemetry + HLS timestamps via Redis to the Apple TV.
 * Zero server-side rendering.
 */

import { useState, useCallback } from 'react';

export default function PushButton({ insightIds, onPush }) {
    const [isPushing, setIsPushing] = useState(false);
    const [pushResult, setPushResult] = useState(null);

    const handlePush = useCallback(async () => {
        if (!insightIds || insightIds.length === 0) return;

        setIsPushing(true);
        setPushResult(null);

        try {
            const res = await fetch('/api/push', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('cruyff_token')}`,
                },
                body: JSON.stringify({
                    insight_ids: insightIds,
                    ttl_minutes: 15,
                }),
            });

            if (res.ok) {
                const data = await res.json();
                setPushResult({
                    success: true,
                    count: data.insights?.length || insightIds.length,
                    push_id: data.push_id,
                });
                onPush?.(data);
            } else {
                setPushResult({ success: false, error: res.statusText });
            }
        } catch (err) {
            setPushResult({ success: false, error: err.message });
        } finally {
            setIsPushing(false);
        }
    }, [insightIds, onPush]);

    return (
        <div className="push-container">
            <button
                className={`push-button ${isPushing ? 'pushing' : ''} ${pushResult?.success ? 'success' : ''}`}
                onClick={handlePush}
                disabled={isPushing || !insightIds?.length}
                id="push-to-tv-button"
            >
                <span className="push-icon">
                    {isPushing ? '📡' : pushResult?.success ? '✅' : '⬆'}
                </span>
                <span className="push-label">
                    {isPushing
                        ? 'Pushing…'
                        : pushResult?.success
                            ? `Sent ${pushResult.count} insights`
                            : `PUSH (${insightIds?.length || 0})`
                    }
                </span>
            </button>

            {pushResult && !pushResult.success && (
                <div className="push-error fade-in">
                    ⚠ {pushResult.error}
                </div>
            )}
        </div>
    );
}
