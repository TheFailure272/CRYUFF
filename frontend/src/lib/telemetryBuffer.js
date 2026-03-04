/**
 * C.R.U.Y.F.F. — Telemetry Ring Buffer (Fix F6)
 *
 * Stores the last N seconds of analysis frames indexed by timestamp.
 * The useFrame loop queries this buffer using the <video>.currentTime
 * to retrieve the frame matching the broadcast timecode — eliminating
 * the 2-10s desync between live WebSocket telemetry and HLS video.
 *
 * Also provides a "live mode" (no video) where it just returns the
 * latest frame.
 */

/**
 * Fixed-size circular buffer for telemetry frames.
 *
 * @param {number} maxFrames - Maximum frames to store (default: 750 = 30s @ 25Hz)
 */
export class TelemetryRingBuffer {
    constructor(maxFrames = 750) {
        this._buffer = new Array(maxFrames);
        this._head = 0;
        this._count = 0;
        this._maxFrames = maxFrames;
        this._latest = null;
    }

    /**
     * Push a new telemetry frame into the buffer.
     * @param {Object} frame - The analysis result with a .timestamp field
     */
    push(frame) {
        if (!frame || typeof frame.timestamp !== 'number') return;
        this._buffer[this._head] = frame;
        this._head = (this._head + 1) % this._maxFrames;
        if (this._count < this._maxFrames) this._count++;
        this._latest = frame;
    }

    /**
     * Get the latest frame (for live mode / no video).
     * @returns {Object|null}
     */
    getLatest() {
        return this._latest;
    }

    /**
     * Query the frame closest to a given timestamp.
     * Uses binary search on the sorted circular buffer.
     *
     * @param {number} targetTime - The video's current timecode (seconds)
     * @returns {Object|null} The closest matching telemetry frame
     */
    getAtTime(targetTime) {
        if (this._count === 0) return null;

        let bestFrame = null;
        let bestDelta = Infinity;

        // Linear scan (buffer is small enough: max 750 entries)
        // For production with larger buffers, use binary search.
        const start = (this._head - this._count + this._maxFrames) % this._maxFrames;
        for (let i = 0; i < this._count; i++) {
            const idx = (start + i) % this._maxFrames;
            const frame = this._buffer[idx];
            if (!frame) continue;
            const delta = Math.abs(frame.timestamp - targetTime);
            if (delta < bestDelta) {
                bestDelta = delta;
                bestFrame = frame;
            }
        }
        return bestFrame;
    }

    /**
     * Get the frame for the current render context.
     *
     * @param {HTMLVideoElement|null} videoEl - The broadcast video element
     * @returns {Object|null} The appropriate telemetry frame
     */
    getForRender(videoEl) {
        if (videoEl && !videoEl.paused && videoEl.currentTime > 0) {
            // AR mode: sync to video timecode
            return this.getAtTime(videoEl.currentTime);
        }
        // Live mode: return latest
        return this._latest;
    }

    get count() {
        return this._count;
    }
}
