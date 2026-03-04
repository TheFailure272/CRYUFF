/**
 * C.R.U.Y.F.F. — BallLayer (Fix F21 + Fix F24 + Fix F27)
 *
 * Fix F24: Cubic Hermite Spline interpolation (velocity-aware).
 *
 * Fix F27: Predictive extrapolation. The optical tracking provider
 * (Metrica, Hawk-Eye) has ~100-200ms processing latency. Without
 * compensation, the Tactical Glass shows the pitch as it looked
 * 150ms ago — the "Ghost of the Present."
 *
 * Solution: When t > 1.0 (past the latest frame), extrapolate
 * forward using the latest velocity vector:
 *   P_now = P1 + V1 × (t - 1) × Δt
 * Clamped to MAX_EXTRAPOLATION (200ms) to prevent runaway prediction.
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const BALL_RADIUS = 0.006;
const BALL_Z = 0.25;
const SHADOW_Z = 0.19;
const FRAME_INTERVAL = 1 / 25; // 25Hz backend
// Fix F27: Max forward extrapolation (200ms — typical provider latency)
const MAX_EXTRAPOLATION = 0.200;

const COLOR_BALL = new THREE.Color(0xffffff);
const COLOR_SHADOW = new THREE.Color(0x000000);

/**
 * Cubic Hermite basis functions.
 *
 * @param {number} t - Interpolation parameter [0,1]
 * @returns {{ h00: number, h10: number, h01: number, h11: number }}
 */
function hermiteBasis(t) {
    const t2 = t * t;
    const t3 = t2 * t;
    return {
        h00: 2 * t3 - 3 * t2 + 1,    // position weight for P0
        h10: t3 - 2 * t2 + t,        // tangent weight for V0
        h01: -2 * t3 + 3 * t2,       // position weight for P1
        h11: t3 - t2,                 // tangent weight for V1
    };
}

/**
 * Evaluate cubic Hermite spline for a single axis.
 */
function hermite1D(t, p0, v0, p1, v1) {
    const { h00, h10, h01, h11 } = hermiteBasis(t);
    return h00 * p0 + h10 * v0 * FRAME_INTERVAL
        + h01 * p1 + h11 * v1 * FRAME_INTERVAL;
}

export default function BallLayer({ dataRef }) {
    const ballRef = useRef();
    const shadowRef = useRef();

    // Store two consecutive frames for Hermite interpolation
    const splineState = useRef({
        // Previous frame
        p0: { x: 0.5, y: 0.5, z: 0 },
        v0: { x: 0, y: 0, z: 0 },
        // Current target frame
        p1: { x: 0.5, y: 0.5, z: 0 },
        v1: { x: 0, y: 0, z: 0 },
        // Timestamp tracking
        lastTimestamp: 0,
        frameStartTime: 0,
        hasData: false,
    });

    const ballGeo = useMemo(() => new THREE.CircleGeometry(BALL_RADIUS, 16), []);
    const shadowGeo = useMemo(() => new THREE.RingGeometry(
        BALL_RADIUS * 0.8, BALL_RADIUS * 1.8, 16
    ), []);

    const ballMat = useMemo(() => new THREE.MeshBasicMaterial({
        color: COLOR_BALL,
        transparent: true,
        depthTest: false,
        opacity: 0.95,
    }), []);

    const shadowMat = useMemo(() => new THREE.MeshBasicMaterial({
        color: COLOR_SHADOW,
        transparent: true,
        depthTest: false,
        opacity: 0.3,
    }), []);

    useFrame(({ clock }) => {
        const ball = ballRef.current;
        const shadow = shadowRef.current;
        if (!ball || !shadow) return;

        const data = dataRef.current;
        const coords = data?.coordinates;
        const timestamp = data?.timestamp || 0;
        const state = splineState.current;

        // Extract ball coords from indices 44-46
        let bx = 0.5, by = 0.5, bz = 0;
        let hasBall = false;
        if (coords && coords.length >= 47) {
            bx = coords[44];
            by = coords[45];
            bz = coords[46];
            hasBall = true;
        }

        // New frame arrived — shift the spline window
        if (timestamp > state.lastTimestamp && hasBall) {
            // P0 ← P1 (previous target becomes previous position)
            state.p0.x = state.p1.x;
            state.p0.y = state.p1.y;
            state.p0.z = state.p1.z;
            state.v0.x = state.v1.x;
            state.v0.y = state.v1.y;
            state.v0.z = state.v1.z;

            // Compute velocity from position delta (m/frame → m/s)
            const dx = bx - state.p0.x;
            const dy = by - state.p0.y;
            const dz = bz - state.p0.z;
            state.v1.x = dx / FRAME_INTERVAL;
            state.v1.y = dy / FRAME_INTERVAL;
            state.v1.z = dz / FRAME_INTERVAL;

            // P1 ← new position
            state.p1.x = bx;
            state.p1.y = by;
            state.p1.z = bz;

            state.lastTimestamp = timestamp;
            state.frameStartTime = clock.getElapsedTime();
            state.hasData = true;
        }

        if (!state.hasData) {
            ball.visible = false;
            shadow.visible = false;
            return;
        }

        // t: how far we are between the last two frames
        // Fix F27: Allow t > 1.0 for predictive extrapolation
        const elapsed = clock.getElapsedTime() - state.frameStartTime;
        const maxT = 1.0 + (MAX_EXTRAPOLATION / FRAME_INTERVAL);
        const t = Math.min(elapsed / FRAME_INTERVAL, maxT);

        let ix, iy, iz;

        if (t <= 1.0) {
            // Standard Hermite interpolation between P0 and P1
            ix = hermite1D(t, state.p0.x, state.v0.x, state.p1.x, state.v1.x);
            iy = hermite1D(t, state.p0.y, state.v0.y, state.p1.y, state.v1.y);
            iz = hermite1D(t, state.p0.z, state.v0.z, state.p1.z, state.v1.z);
        } else {
            // Fix F27: Predictive extrapolation past P1
            // P_now = P1 + V1 × (t - 1) × Δt
            const overshoot = (t - 1.0) * FRAME_INTERVAL;
            ix = state.p1.x + state.v1.x * overshoot;
            iy = state.p1.y + state.v1.y * overshoot;
            iz = state.p1.z + state.v1.z * overshoot;
        }
        // Fix F31: Goal-net collision (replaces global pitch clamp)
        // A football pitch has no invisible walls — balls go out of bounds.
        // Only the physical goal nets stop the ball.
        //
        // Goal geometry (normalised coordinates):
        //   Goal 1: Y=0 plane, X ∈ [0.37, 0.63], Z ∈ [0, 0.036]
        //   Goal 2: Y=1 plane, X ∈ [0.37, 0.63], Z ∈ [0, 0.036]
        //   (7.32m posts / 105m pitch ≈ 0.035 each side of center)
        //   (2.44m crossbar / 68m ≈ 0.036 normalised height)
        if (t > 1.0) {
            const GOAL_X_MIN = 0.5 - 0.13;  // left post
            const GOAL_X_MAX = 0.5 + 0.13;  // right post
            const GOAL_Z_MAX = 0.036;        // crossbar height (normalised)

            // Check goal 1 (Y=0): ball moving toward Y=0?
            if (state.v1.y < 0 && iy <= 0) {
                // Ray-plane: did the velocity cross Y=0?
                if (ix >= GOAL_X_MIN && ix <= GOAL_X_MAX && iz >= 0 && iz <= GOAL_Z_MAX) {
                    // Ball hit the net — clamp at goal line
                    iy = 0;
                }
                // else: ball went wide/over — let it fly out of bounds
            }

            // Check goal 2 (Y=1): ball moving toward Y=1?
            if (state.v1.y > 0 && iy >= 1) {
                if (ix >= GOAL_X_MIN && ix <= GOAL_X_MAX && iz >= 0 && iz <= GOAL_Z_MAX) {
                    iy = 1;
                }
            }
        }

        // Z floor only — ball cannot go underground
        iz = Math.max(0, iz);

        // Ball position (raised slightly when airborne)
        const airOffset = Math.min(Math.max(iz, 0) * 0.02, 0.04);
        ball.position.set(ix, iy + airOffset, BALL_Z);
        ball.visible = true;

        // Scale up when airborne
        const scaleBoost = 1 + Math.max(iz, 0) * 0.1;
        ball.scale.set(scaleBoost, scaleBoost, 1);

        // Shadow when airborne
        const airborne = iz > 0.3;
        shadow.visible = airborne;
        if (airborne) {
            shadow.position.set(ix, iy, SHADOW_Z);
            const shadowScale = 1 + iz * 0.15;
            shadow.scale.set(shadowScale, shadowScale, 1);
            shadowMat.opacity = Math.min(0.4, iz * 0.1);
        }
    });

    return (
        <group renderOrder={2}>
            <mesh ref={shadowRef} geometry={shadowGeo} material={shadowMat} visible={false} />
            <mesh ref={ballRef} geometry={ballGeo} material={ballMat} />
        </group>
    );
}
