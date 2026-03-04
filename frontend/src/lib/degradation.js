/**
 * C.R.U.Y.F.F. — Battery-Aware Degradation Engine (Fix F16 + Fix F18)
 *
 * Fix F18: Safari/iOS removed navigator.getBattery() for privacy.
 * Instead of relying on the Battery API, we build a Thermal Inference
 * Heuristic that measures useFrame render time via performance.now().
 *
 * If the render loop consistently exceeds 16ms (missing 60fps),
 * the device is thermally throttling — force REDUCED or MINIMAL mode.
 *
 * Dual-source strategy:
 *   1. navigator.getBattery() — used on Android/Chrome where available
 *   2. Frame-drop heuristic — universal fallback (especially iOS/Safari)
 *
 * Tiers:
 *   FULL    (>30% battery OR <5% frame drops)
 *   REDUCED (≤30% battery OR 5-15% frame drops)
 *   MINIMAL (≤15% battery OR >15% frame drops)
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/** @enum {string} */
export const DegradationLevel = {
    FULL: 'full',
    REDUCED: 'reduced',
    MINIMAL: 'minimal',
};

/**
 * Sliding-window frame time tracker.
 * Tracks the percentage of frames exceeding a threshold over the last N frames.
 */
class FrameDropTracker {
    constructor(windowSize = 120) {  // ~2 seconds @ 60fps
        this._window = new Float32Array(windowSize);
        this._head = 0;
        this._count = 0;
        this._size = windowSize;
        this._threshold = 16.67;  // 60fps target (ms)
    }

    /**
     * Record a frame's render time.
     * @param {number} ms - Frame render time in milliseconds
     */
    push(ms) {
        this._window[this._head] = ms;
        this._head = (this._head + 1) % this._size;
        if (this._count < this._size) this._count++;
    }

    /**
     * Get the percentage of frames that exceeded the threshold.
     * @returns {number} 0.0 to 1.0
     */
    get dropRate() {
        if (this._count === 0) return 0;
        let drops = 0;
        const start = (this._head - this._count + this._size) % this._size;
        for (let i = 0; i < this._count; i++) {
            const idx = (start + i) % this._size;
            if (this._window[idx] > this._threshold) drops++;
        }
        return drops / this._count;
    }
}

/**
 * Compute degradation level from drop rate.
 * @param {number} dropRate - 0.0 to 1.0
 * @returns {string}
 */
function levelFromDropRate(dropRate) {
    if (dropRate > 0.15) return DegradationLevel.MINIMAL;
    if (dropRate > 0.05) return DegradationLevel.REDUCED;
    return DegradationLevel.FULL;
}

/**
 * Build degradation state from level.
 * @param {string} level
 * @param {number} battery - 0-1 (or -1 if unknown)
 * @param {boolean} charging
 * @returns {Object}
 */
function buildState(level, battery, charging) {
    switch (level) {
        case DegradationLevel.MINIMAL:
            return {
                level, battery, charging,
                useHighpShader: false,
                enableBreathing: false,
                bufferSize: 150,
                dpr: 0.75,    // Fix F19: sub-native resolution, max GPU savings
            };
        case DegradationLevel.REDUCED:
            return {
                level, battery, charging,
                useHighpShader: true,
                enableBreathing: false,
                bufferSize: 375,
                dpr: 1,       // Fix F19: 1:1 CSS pixel, no Retina overhead
            };
        default:
            return {
                level: DegradationLevel.FULL, battery, charging,
                useHighpShader: true,
                enableBreathing: true,
                bufferSize: 750,
                dpr: Math.min(window.devicePixelRatio || 2, 1.5), // Fix F19: cap Retina
            };
    }
}

/**
 * React hook: monitors battery (where available) AND frame timing.
 * The worse of the two sources wins — if EITHER detects stress, degrade.
 *
 * @returns {Object} DegradationState + reportFrameTime function
 */
export function useDegradation() {
    const [state, setState] = useState(
        buildState(DegradationLevel.FULL, 1.0, true)
    );

    const trackerRef = useRef(new FrameDropTracker(120));
    const batteryLevelRef = useRef({ level: 1.0, charging: true });
    const intervalRef = useRef(null);

    // ── Source 1: Battery API (Android/Chrome only) ──────────
    useEffect(() => {
        let mounted = true;

        if (typeof navigator !== 'undefined' && 'getBattery' in navigator) {
            navigator.getBattery().then((batt) => {
                if (!mounted) return;

                const sync = () => {
                    batteryLevelRef.current = {
                        level: batt.level,
                        charging: batt.charging,
                    };
                };
                sync();
                batt.addEventListener('levelchange', sync);
                batt.addEventListener('chargingchange', sync);
            }).catch(() => {
                // getBattery() rejected (Firefox) — rely on heuristic only
            });
        }

        return () => { mounted = false; };
    }, []);

    // ── Source 2: Frame-drop thermal heuristic (universal) ────
    useEffect(() => {
        // Check every 2 seconds
        intervalRef.current = setInterval(() => {
            const tracker = trackerRef.current;
            const dropRate = tracker.dropRate;
            const thermalLevel = levelFromDropRate(dropRate);

            const { level: battLevel, charging } = batteryLevelRef.current;

            // Battery-based level
            let batteryDegLevel = DegradationLevel.FULL;
            if (!charging) {
                if (battLevel <= 0.15) batteryDegLevel = DegradationLevel.MINIMAL;
                else if (battLevel <= 0.30) batteryDegLevel = DegradationLevel.REDUCED;
            }

            // The WORSE of the two wins
            const levels = [DegradationLevel.FULL, DegradationLevel.REDUCED, DegradationLevel.MINIMAL];
            const finalIdx = Math.max(levels.indexOf(thermalLevel), levels.indexOf(batteryDegLevel));
            const finalLevel = levels[finalIdx];

            setState(buildState(finalLevel, battLevel, charging));
        }, 2000);

        return () => clearInterval(intervalRef.current);
    }, []);

    /**
     * Call this from useFrame with each frame's render time.
     * @param {number} ms - render duration in milliseconds
     */
    const reportFrameTime = useCallback((ms) => {
        trackerRef.current.push(ms);
    }, []);

    return { ...state, reportFrameTime };
}
