/**
 * C.R.U.Y.F.F. — WebGL Context Recovery (Fix F14)
 *
 * Handles the iOS/iPadOS guillotine: when the manager swipes away
 * during half-time, iOS dumps GPU memory. On return, the WebGL
 * context is lost and the screen goes black.
 *
 * This hook:
 * 1. Listens for `webglcontextlost` — prevents default (allows restore)
 * 2. Listens for `webglcontextrestored` — triggers full re-init:
 *    - Re-uploads InstancedMesh matrices
 *    - Re-compiles ShaderMaterial programs
 *    - Re-establishes the WebSocket stream
 * 3. Exposes `contextLost` state for HUD to show a "RECOVERING…" banner
 */

import { useEffect, useState, useCallback } from 'react';
import { useThree } from '@react-three/fiber';

/**
 * Must be used inside the R3F Canvas tree.
 *
 * @param {Object} opts
 * @param {Function} opts.onRestore - callback to trigger full scene re-init
 * @returns {{ contextLost: boolean }}
 */
export function useContextRecovery({ onRestore } = {}) {
    const { gl } = useThree();
    const [contextLost, setContextLost] = useState(false);

    const handleLost = useCallback((event) => {
        // Prevent default allows the browser to attempt restoration
        event.preventDefault();
        setContextLost(true);
        console.warn('[C.R.U.Y.F.F.] WebGL context lost — awaiting restoration');
    }, []);

    const handleRestored = useCallback(() => {
        console.info('[C.R.U.Y.F.F.] WebGL context restored — re-initializing');
        setContextLost(false);

        // Force Three.js to re-upload all GPU resources
        const renderer = gl;
        if (renderer) {
            // Reset internal state so Three.js knows to re-upload everything
            renderer.state.reset();

            // Force texture re-upload by clearing internal cache
            if (renderer.properties) {
                renderer.properties.dispose();
            }
        }

        // Notify parent to re-init InstancedMesh matrices etc.
        onRestore?.();
    }, [gl, onRestore]);

    useEffect(() => {
        const canvas = gl?.domElement;
        if (!canvas) return;

        canvas.addEventListener('webglcontextlost', handleLost);
        canvas.addEventListener('webglcontextrestored', handleRestored);

        return () => {
            canvas.removeEventListener('webglcontextlost', handleLost);
            canvas.removeEventListener('webglcontextrestored', handleRestored);
        };
    }, [gl, handleLost, handleRestored]);

    return { contextLost };
}
