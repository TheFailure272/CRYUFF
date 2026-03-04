/**
 * C.R.U.Y.F.F. — useVideoSync (Fix F10)
 *
 * Uses `requestVideoFrameCallback` (rVFC) to capture the exact timestamp
 * when a new video frame is presented to the screen. This eliminates the
 * `video.currentTime` micro-stutter caused by the media decoder updating
 * at 4-15Hz while useFrame polls at 120Hz.
 *
 * Architecture:
 * - rVFC fires once per decoded video frame (typically 25-30Hz)
 * - Captures { mediaTime, performanceNow } at that instant
 * - useFrame reads these cached values + interpolates with performance.now()
 *   to compute sub-frame telemetry time at full display refresh rate
 *
 * Fallback: If rVFC is unavailable (older browsers), falls back to
 * `currentTime` polling with a smoothing filter.
 */

import { useRef, useEffect, useCallback } from 'react';

/**
 * @param {React.RefObject<HTMLVideoElement>} videoRef - ref to the <video> element
 * @returns {{ getVideoTime: () => number, isActive: boolean }}
 */
export function useVideoSync(videoRef) {
    // Last rVFC sample: { mediaTime, perfNow }
    const lastSample = useRef({ mediaTime: 0, perfNow: 0 });
    const activeRef = useRef(false);
    const rVFCSupported = useRef(false);
    const callbackIdRef = useRef(null);

    useEffect(() => {
        const video = videoRef?.current;
        if (!video) return;

        // Feature detection
        rVFCSupported.current = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

        if (rVFCSupported.current) {
            const onVideoFrame = (now, metadata) => {
                lastSample.current = {
                    mediaTime: metadata.mediaTime,
                    perfNow: now,
                };
                activeRef.current = true;
                // Re-register for next frame
                callbackIdRef.current = video.requestVideoFrameCallback(onVideoFrame);
            };

            callbackIdRef.current = video.requestVideoFrameCallback(onVideoFrame);

            return () => {
                if (callbackIdRef.current !== null) {
                    video.cancelVideoFrameCallback(callbackIdRef.current);
                }
            };
        } else {
            // Fallback: poll currentTime at ~15Hz
            const intervalId = setInterval(() => {
                if (!video.paused && video.currentTime > 0) {
                    lastSample.current = {
                        mediaTime: video.currentTime,
                        perfNow: performance.now(),
                    };
                    activeRef.current = true;
                }
            }, 66); // ~15Hz

            return () => clearInterval(intervalId);
        }
    }, [videoRef]);

    /**
     * Get the interpolated video time for the current render frame.
     * Called from useFrame at 60-120Hz.
     *
     * Uses the last rVFC sample and micro-interpolates using
     * the elapsed time since that sample was captured.
     *
     * @returns {number} The estimated video media time in seconds
     */
    const getVideoTime = useCallback(() => {
        const sample = lastSample.current;
        if (!activeRef.current || sample.perfNow === 0) return 0;

        // Micro-interpolation: how long since the last rVFC sample?
        const elapsed = (performance.now() - sample.perfNow) / 1000;
        return sample.mediaTime + elapsed;
    }, []);

    return {
        getVideoTime,
        isActive: activeRef.current,
        isRVFCSupported: rVFCSupported.current,
    };
}
