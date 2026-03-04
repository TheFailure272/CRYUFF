/**
 * C.R.U.Y.F.F. — PitchCanvas (Final Deployment-Hardened)
 *
 * Fixes integrated:
 * - Fix F3:  Canvas alpha: true (AR video overlay)
 * - Fix F4:  Strict Z-layering (child render order)
 * - Fix F13: PTZ Homography (FRI-damped TRS with SLERP)
 * - Fix F14: WebGL context loss/recovery (iOS half-time swipe)
 * - Fix F15: Video-bound render architecture
 * - Fix F16: Battery degradation awareness (passed to layers)
 * - Fix F17: Matrix decomposition (TRS + SLERP, not raw LERP)
 * - Fix F18: Thermal inference heuristic (FrameTimeReporter)
 */

import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { useEffect, useRef, useCallback } from 'react';
import { COLORS, PITCH } from '../lib/constants';
import { PTZHomography } from '../lib/homography';
import { useContextRecovery } from '../hooks/useContextRecovery';
import PitchLines from './PitchLines';
import FlowLayer from './FlowLayer';
import PulseLayer from './PulseLayer';
import GhostLayer from './GhostLayer';
import BallLayer from './BallLayer';
import SetPieceOverlay from './SetPieceOverlay';

/** Fits orthographic camera to viewport (demo mode). */
function CameraFit() {
    const { camera, size } = useThree();

    useEffect(() => {
        const viewportAspect = size.width / size.height;
        const pitchAspect = PITCH.ASPECT;
        const padding = 1.15;

        if (viewportAspect > pitchAspect) {
            camera.zoom = size.height / padding;
        } else {
            camera.zoom = size.width / (padding * pitchAspect);
        }
        camera.position.set(0.5, 0.5, 10);
        camera.updateProjectionMatrix();
    }, [camera, size]);

    return null;
}

/**
 * Fix F13 + F17: Reads camera_matrix from telemetry, decomposes into
 * T/R/S, SLERPs the rotation quaternion, FRI-damps position & scale.
 */
function PTZController({ dataRef, homography }) {
    const { camera } = useThree();

    useFrame((_, delta) => {
        const data = dataRef.current;
        homography.update(data);

        if (homography.isActive) {
            homography.applyToCamera(camera, delta);
        }
    });

    return null;
}

/**
 * Fix F14: WebGL context recovery orchestrator.
 */
function ContextGuard({ onContextChange }) {
    const { contextLost } = useContextRecovery({
        onRestore: () => {
            onContextChange?.(false);
        },
    });

    useEffect(() => {
        if (contextLost) onContextChange?.(true);
    }, [contextLost, onContextChange]);

    return null;
}

/**
 * Fix F18: Measures useFrame render time and feeds to the degradation
 * engine's thermal inference heuristic. iOS-safe replacement for getBattery().
 */
function FrameTimeReporter({ reportFrameTime }) {
    const lastRef = useRef(performance.now());

    useFrame(() => {
        const now = performance.now();
        const dt = now - lastRef.current;
        lastRef.current = now;
        reportFrameTime?.(dt);
    });

    return null;
}

export default function PitchCanvas({ dataRef, degradation }) {
    const homographyRef = useRef(new PTZHomography());

    const handleContextChange = useCallback((lost) => {
        if (lost) {
            console.warn('[C.R.U.Y.F.F.] Context lost — render paused');
        } else {
            console.info('[C.R.U.Y.F.F.] Context restored — scene rebuilt');
        }
    }, []);

    return (
        <Canvas
            className="canvas-layer"
            gl={{
                alpha: true,
                antialias: true,
                powerPreference: 'high-performance',
            }}
            orthographic
            camera={{
                zoom: 1,
                position: [0.5, 0.5, 10],
                near: 0.1,
                far: 100,
            }}
            dpr={degradation?.dpr || Math.min(window.devicePixelRatio, 1.5)}
            frameloop="always"
        >
            <color attach="background" args={[COLORS.PITCH_BG]} />
            <ambientLight intensity={1} />

            {/* Infrastructure */}
            <CameraFit />
            <PTZController dataRef={dataRef} homography={homographyRef.current} />
            <ContextGuard onContextChange={handleContextChange} />
            <FrameTimeReporter reportFrameTime={degradation?.reportFrameTime} />

            {/* Render layers (Z-ordered by renderOrder) */}
            <PitchLines />
            <FlowLayer dataRef={dataRef} degradation={degradation} />
            <PulseLayer dataRef={dataRef} />
            <BallLayer dataRef={dataRef} />
            <GhostLayer dataRef={dataRef} />
            <SetPieceOverlay dataRef={dataRef} />
        </Canvas>
    );
}
