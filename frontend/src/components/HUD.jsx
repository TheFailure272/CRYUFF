/**
 * C.R.U.Y.F.F. — HUD (Heads-Up Display)
 *
 * Glassmorphism sidebar with:
 * - Connection status (LIVE / RECONNECTING / OFFLINE)
 * - Void summary (count + max persistence)
 * - Bio alert list (amber/red players)
 * - Ghost panel (Optimize button + xT result)
 *
 * This component uses React state (hudState) which updates at 4Hz —
 * it does NOT need 25Hz updates because text doesn't need frame-level
 * precision.
 */

import OptimizeButton from './OptimizeButton';

export default function HUD({ hudState, status, requestGhost }) {
    const {
        voidCount,
        maxPersistence,
        bioAlerts,
        ghostXt,
    } = hudState;

    return (
        <div className="hud">
            {/* ── Connection Status ─────────────── */}
            <div className="hud-panel fade-in">
                <div className="hud-title">System</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`status-dot ${status === 'connected' ? 'live' : status === 'reconnecting' ? 'reconnecting' : 'disconnected'}`} />
                    <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                        {status === 'connected' ? 'LIVE' : status === 'reconnecting' ? 'RECONNECTING…' : 'OFFLINE'}
                    </span>
                </div>
                <div className="label" style={{ marginTop: 8 }}>
                    C.R.U.Y.F.F. v1.0 · 25Hz
                </div>
            </div>

            {/* ── The Flow (Topology) ─────────────── */}
            <div className="hud-panel fade-in" style={{ animationDelay: '0.05s' }}>
                <div className="hud-title">⬡ The Flow</div>
                <div className="void-metric">
                    <span className="value">{voidCount}</span>
                    <span className="unit">stable voids</span>
                </div>
                {maxPersistence > 0 && (
                    <div style={{ marginTop: 4, fontSize: '0.7rem', color: '#8888aa' }}>
                        max persistence: <span className="mono" style={{ color: '#00f5d4' }}>{maxPersistence.toFixed(3)}</span>
                    </div>
                )}
            </div>

            {/* ── The Pulse (Bio-Kinetics) ──────── */}
            <div className="hud-panel fade-in" style={{ animationDelay: '0.1s', flex: 1, overflowY: 'auto' }}>
                <div className="hud-title">♥ The Pulse</div>
                {bioAlerts.length === 0 ? (
                    <div style={{ fontSize: '0.75rem', color: '#555566' }}>
                        All players nominal
                    </div>
                ) : (
                    bioAlerts.map((alert) => (
                        <div key={alert.player_id} className="bio-alert-row">
                            <span className={`bio-badge ${alert.status}`}>
                                {alert.player_id}
                            </span>
                            <div className="bio-info">
                                <strong>Player {alert.player_id}</strong>
                                {alert.type === 'structural_collapse' && (
                                    <span style={{ color: '#ef4444', fontSize: '0.6rem', marginLeft: 6 }}>
                                        ⚠ COLLAPSE
                                    </span>
                                )}
                                {alert.type === 'masking' && (
                                    <span style={{ color: '#fbbf24', fontSize: '0.6rem', marginLeft: 6 }}>
                                        ◉ MASKING
                                    </span>
                                )}
                                {alert.type === 'overload' && (
                                    <span style={{ color: '#fbbf24', fontSize: '0.6rem', marginLeft: 6 }}>
                                        ↑ OVERLOAD
                                    </span>
                                )}
                                <br />
                                {alert.hr_bpm ? (
                                    <>
                                        HR: <span className="mono" style={{ color: alert.hr_max_pct > 0.9 ? '#ef4444' : '#fbbf24' }}>
                                            {Math.round(alert.hr_bpm)}
                                        </span>bpm ({(alert.hr_max_pct * 100).toFixed(0)}%) ·
                                        MP: <span className="mono">{alert.metabolic_power?.toFixed(1)}</span>W/kg
                                    </>
                                ) : (
                                    <>
                                        scan: <span className="mono">{alert.scan_frequency?.toFixed(1)}</span>Hz ·
                                        μvar: <span className="mono">{alert.micro_variance?.toFixed(3)}</span>
                                    </>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* ── The Ghost (Trajectory) ──────────── */}
            <div className="hud-panel fade-in" style={{ animationDelay: '0.15s' }}>
                <div className="hud-title">⚡ The Ghost</div>
                <OptimizeButton requestGhost={requestGhost} />
                {ghostXt !== null && (
                    <div className="xt-result">
                        <span className="value">{ghostXt.toFixed(2)}</span>
                        <span className="unit">xT gained</span>
                    </div>
                )}
            </div>

            {/* ── System Tag ──────────────────────── */}
            <div className="system-tag">
                C.R.U.Y.F.F. · Tactical Glass
            </div>
        </div>
    );
}
