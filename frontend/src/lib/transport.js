/**
 * C.R.U.Y.F.F. — Transport Layer (Final)
 *
 * Fixes applied:
 * - Fix F6: TelemetryRingBuffer integration
 * - Fix F9: Selective throttle bypass
 * - Fix F23: JWT auth over wss:// (zero-trust dugout security)
 *
 * Token lifecycle:
 *   1. Club IT generates a JWT pre-match via auth.py
 *   2. Token is stored in localStorage('cruyff_token')
 *   3. Transport appends it as ?token= query param on wss://
 *   4. Gateway validates before upgrading the connection
 */

import { TRANSPORT } from './constants';

/**
 * @param {Object} opts
 * @param {React.MutableRefObject} opts.dataRef - latest frame for useFrame
 * @param {import('./telemetryBuffer').TelemetryRingBuffer} opts.ringBuffer - for video sync
 * @param {Function} opts.onHudUpdate - throttled HUD callback (4Hz)
 * @param {Function} opts.onCriticalAlert - IMMEDIATE callback for red alerts / ghost results
 * @param {Function} opts.onStatusChange - connection status
 * @returns {{ send: Function, close: Function }}
 */
export function createAnalysisStream({
    dataRef,
    ringBuffer,
    onHudUpdate,
    onCriticalAlert,
    onStatusChange,
}) {
    let ws = null;
    let attempt = 0;
    let closed = false;
    let lastMono = 0;
    let hudThrottleId = null;
    const hudInterval = 1000 / TRANSPORT.HUD_UPDATE_HZ;

    // Track previous state for critical change detection
    let prevRedCount = 0;
    let prevGhostXt = null;
    let wsOpenTime = 0; // Fix F30: tracks connection start for 1006 detection

    async function connect() {
        if (closed) return;

        // Fix F26: Ticket-based handshake (OWASP-safe)
        // Step 1: Exchange JWT for a 5-second ephemeral ticket via POST
        // The JWT travels in the Authorization header (encrypted by TLS),
        // NEVER in the URL where it would be logged by proxies.
        const token = localStorage.getItem('cruyff_token') || '';
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsUrl = `${proto}//${location.host}/ws/tracking`;

        if (token) {
            try {
                const res = await fetch('/ticket', {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` },
                });
                if (res.ok) {
                    const { ticket } = await res.json();
                    // The ticket is opaque, one-time-use, expires in 5s
                    wsUrl += `?ticket=${encodeURIComponent(ticket)}`;
                } else {
                    console.error('[C.R.U.Y.F.F.] Ticket exchange failed:', res.status);
                }
            } catch (e) {
                // In dev mode, connect without auth
                console.warn('[C.R.U.Y.F.F.] Ticket endpoint unavailable (dev mode)');
            }
        }

        wsOpenTime = performance.now(); // Fix F30: mark connection start
        ws = new WebSocket(wsUrl);
        onStatusChange?.('reconnecting');

        ws.onopen = () => {
            attempt = 0;
            onStatusChange?.('connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Skip gateway protocol messages
                if (data.type === 'reconnect_policy' || data.type === 'transport_hysteresis') return;

                // Frame dedup by monotonic timestamp
                const mono = data.timestamp || 0;
                if (mono > 0 && mono <= lastMono) return;
                lastMono = mono;

                // Push into ring buffer (Fix F6 — for video sync)
                ringBuffer?.push(data);

                // Update mutable ref (for live mode useFrame reads)
                dataRef.current = data;

                // ── Fix F9: Selective Throttle Bypass ────────────────
                const bioAlerts = data.bio_alerts || [];
                const redAlerts = bioAlerts.filter((a) => a.status === 'red');
                const ghostXt = data.ghost?.expected_xt ?? null;

                // Critical: new red alert appeared
                const redCountChanged = redAlerts.length !== prevRedCount;
                // Critical: ghost result arrived
                const ghostChanged = ghostXt !== null && ghostXt !== prevGhostXt;

                if (redCountChanged || ghostChanged) {
                    prevRedCount = redAlerts.length;
                    prevGhostXt = ghostXt;
                    // Bypass throttle — immediate React update
                    onCriticalAlert?.(data);
                }

                // Standard throttled HUD update (4Hz)
                if (!hudThrottleId) {
                    hudThrottleId = setTimeout(() => {
                        hudThrottleId = null;
                        onHudUpdate?.(data);
                    }, hudInterval);
                }
            } catch {
                // malformed JSON
            }
        };

        ws.onclose = (event) => {
            if (closed) return;
            onStatusChange?.('reconnecting');

            // Fix F28: Detect ticket-related close codes
            const isExplicitTicketFailure = event.code === 4401 || event.code === 4403;

            // Fix F30 + F32: Detect proxy-swallowed ticket failure
            // Stadium load balancers strip custom close codes → generic 1006.
            // But a fast 1006 can also mean Wi-Fi dead zone (manager walked
            // behind a concrete pillar). navigator.onLine discriminates:
            //   onLine=true  + fast 1006 → proxy swallowed the ticket code
            //   onLine=false + fast 1006 → Wi-Fi dropped, use fast reconnect
            const connectionAge = performance.now() - wsOpenTime;
            const isProxySwallow = event.code === 1006
                && connectionAge < 1000
                && navigator.onLine; // Fix F32: only if network is physically present

            scheduleReconnect(isExplicitTicketFailure || isProxySwallow);
        };

        ws.onerror = () => {
            ws?.close();
        };
    }

    function scheduleReconnect(ticketFailure = false) {
        attempt += 1;
        const base = TRANSPORT.RECONNECT_BASE_MS;
        const max = TRANSPORT.RECONNECT_MAX_MS;

        // Fix F28: On ticket failure, apply longer backoff to give
        // the network time to stabilize before requesting a new ticket.
        // Standard reconnect: 1s, 2s, 4s...
        // Ticket failure:     2s, 4s, 8s... (double the base)
        const effectiveBase = ticketFailure ? base * 2 : base;
        const delay = Math.min(effectiveBase * Math.pow(2, attempt) + Math.random() * 1000, max);
        setTimeout(connect, delay);
    }

    function send(payload) {
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(payload));
        }
    }

    function close() {
        closed = true;
        if (hudThrottleId) clearTimeout(hudThrottleId);
        ws?.close();
    }

    connect();
    return { send, close };
}
