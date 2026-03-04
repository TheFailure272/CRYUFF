/**
 * C.R.U.Y.F.F. — Touch Kalman Filter (Feature 1)
 *
 * Predictive touch tracking for the Omni-Cam.
 *
 * Problem: WebRTC round-trip (touch → server render → H.264
 * back to tablet) introduces 60-100ms of motion-to-photon latency.
 * Anything above 40ms feels heavy and nauseating.
 *
 * Solution: Lightweight Kalman filter running on the iPad that
 * predicts where the manager's finger WILL BE 80ms into the future.
 * The server renders the predicted view; by the time the frame
 * arrives, it matches the actual finger position.
 *
 * State vector: [x, y, vx, vy]
 * Measurement:  [x, y] (raw touch coordinates)
 * Prediction:   80ms ahead using velocity estimate
 */

const PREDICTION_MS = 80;  // Predict this far ahead
const PROCESS_NOISE = 0.1;
const MEASUREMENT_NOISE = 0.5;

export class TouchKalman {
    constructor() {
        // State: [x, y, vx, vy]
        this.state = [0.5, 0.5, 0, 0];

        // Covariance matrix (4×4 diagonal)
        this.P = [10, 10, 5, 5];

        this.lastTime = null;
        this.initialized = false;
    }

    /**
     * Update with a raw touch event and return predicted position.
     *
     * @param {number} x - Raw touch X (normalised 0-1)
     * @param {number} y - Raw touch Y (normalised 0-1)
     * @param {number} timestamp - Event timestamp (ms)
     * @returns {{ x: number, y: number, vx: number, vy: number }}
     *   Predicted position 80ms into the future
     */
    update(x, y, timestamp) {
        if (!this.initialized || this.lastTime === null) {
            this.state = [x, y, 0, 0];
            this.lastTime = timestamp;
            this.initialized = true;
            return { x, y, vx: 0, vy: 0 };
        }

        const dt = Math.max(0.001, (timestamp - this.lastTime) / 1000);
        this.lastTime = timestamp;

        // ── Predict step ────────────────────────────
        // State transition: x += vx*dt, y += vy*dt
        const px = this.state[0] + this.state[2] * dt;
        const py = this.state[1] + this.state[3] * dt;
        const pvx = this.state[2];
        const pvy = this.state[3];

        // Covariance prediction
        const pP = [
            this.P[0] + dt * dt * this.P[2] + PROCESS_NOISE * dt,
            this.P[1] + dt * dt * this.P[3] + PROCESS_NOISE * dt,
            this.P[2] + PROCESS_NOISE * dt,
            this.P[3] + PROCESS_NOISE * dt,
        ];

        // ── Update step ─────────────────────────────
        // Innovation (measurement residual)
        const ix = x - px;
        const iy = y - py;

        // Innovation covariance
        const sx = pP[0] + MEASUREMENT_NOISE;
        const sy = pP[1] + MEASUREMENT_NOISE;

        // Kalman gain (simplified diagonal)
        const kx = pP[0] / sx;
        const ky = pP[1] / sy;
        const kvx = pP[2] / sx;
        const kvy = pP[3] / sy;

        // Updated state
        this.state[0] = px + kx * ix;
        this.state[1] = py + ky * iy;
        this.state[2] = pvx + kvx * ix;
        this.state[3] = pvy + kvy * iy;

        // Updated covariance
        this.P[0] = (1 - kx) * pP[0];
        this.P[1] = (1 - ky) * pP[1];
        this.P[2] = (1 - kvx) * pP[2];
        this.P[3] = (1 - kvy) * pP[3];

        // ── Predict forward 80ms ────────────────────
        const predDt = PREDICTION_MS / 1000;
        return {
            x: this.state[0] + this.state[2] * predDt,
            y: this.state[1] + this.state[3] * predDt,
            vx: this.state[2],
            vy: this.state[3],
        };
    }

    /**
     * Reset the filter (e.g., when touch starts again).
     */
    reset() {
        this.state = [0.5, 0.5, 0, 0];
        this.P = [10, 10, 5, 5];
        this.lastTime = null;
        this.initialized = false;
    }
}
