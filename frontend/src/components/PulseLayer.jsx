/**
 * C.R.U.Y.F.F. — PulseLayer (Layer 2: Bio-Kinetic Player Dots) — Revised
 *
 * Fixes applied:
 * - Fix F7: Frame-rate independent damping via 1 - exp(-λ * Δt)
 * - Fix F8: InstancedMesh — 22 players rendered in 1 draw call
 *           (was 22 separate meshes = 22 draw calls)
 *
 * Architecture:
 * - Single InstancedMesh with 22 instances
 * - Per-instance color via instanceColor attribute
 * - Per-instance position via instance matrix
 * - Alert ring overlay as a second InstancedMesh
 */

import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Z, COLORS, ANIM } from '../lib/constants';

const PLAYER_COUNT = 22;
const DOT_RADIUS = 0.008;

const COLOR_GREEN = new THREE.Color(COLORS.PULSE_GREEN);
const COLOR_AMBER = new THREE.Color(COLORS.PULSE_AMBER);
const COLOR_RED = new THREE.Color(COLORS.PULSE_RED);
const tmpMatrix = new THREE.Matrix4();
const tmpPosition = new THREE.Vector3();
const tmpScale = new THREE.Vector3(1, 1, 1);
const tmpQuat = new THREE.Quaternion();
const tmpColor = new THREE.Color();

/** FRI damping factor */
function friAlpha(delta) {
    return 1 - Math.exp(-ANIM.DAMP_LAMBDA * delta);
}

export default function PulseLayer({ dataRef }) {
    const dotsRef = useRef();
    const ringsRef = useRef();

    // Track current interpolated positions
    const currentPositions = useRef(
        Array.from({ length: PLAYER_COUNT }, () => ({ x: 0.5, y: 0.5 }))
    );

    const dotGeo = useMemo(() => new THREE.CircleGeometry(DOT_RADIUS, 16), []);
    const ringGeo = useMemo(() => new THREE.RingGeometry(
        DOT_RADIUS * 1.5, DOT_RADIUS * 2.5, 24
    ), []);

    const dotMat = useMemo(() => new THREE.MeshBasicMaterial({
        transparent: true,
        depthTest: false,
        opacity: 0.9,
    }), []);

    const ringMat = useMemo(() => new THREE.MeshBasicMaterial({
        transparent: true,
        depthTest: false,
        side: THREE.DoubleSide,
        opacity: 0.5,
    }), []);

    // Initialize instance colors
    useEffect(() => {
        const dots = dotsRef.current;
        const rings = ringsRef.current;
        if (!dots || !rings) return;

        for (let i = 0; i < PLAYER_COUNT; i++) {
            dots.setColorAt(i, COLOR_GREEN);
            rings.setColorAt(i, COLOR_RED);
            // Initial hidden position
            tmpMatrix.makeTranslation(0.5, 0.5, Z.PULSE);
            dots.setMatrixAt(i, tmpMatrix);
            tmpMatrix.makeTranslation(0.5, 0.5, Z.PULSE - 0.01);
            rings.setMatrixAt(i, tmpMatrix);
        }
        dots.instanceMatrix.needsUpdate = true;
        dots.instanceColor.needsUpdate = true;
        rings.instanceMatrix.needsUpdate = true;
        rings.instanceColor.needsUpdate = true;
    }, []);

    useFrame(({ clock }, delta) => {
        const dots = dotsRef.current;
        const rings = ringsRef.current;
        if (!dots || !rings) return;

        const data = dataRef.current;
        const coords = data?.coordinates;
        const alerts = data?.bio_alerts || [];
        const t = clock.getElapsedTime();
        const alpha = friAlpha(delta); // Fix F7: framerate-independent

        // Build alert lookup
        const alertMap = {};
        for (const a of alerts) {
            alertMap[a.player_id] = a.status;
        }

        let dotsColorDirty = false;
        let ringsColorDirty = false;

        for (let i = 0; i < PLAYER_COUNT; i++) {
            const cur = currentPositions.current[i];

            // Target position from flat coordinate array
            let tx = 0.5, ty = 0.5;
            let visible = true;
            if (coords && coords.length >= 44) {
                tx = coords[i * 2];
                ty = coords[i * 2 + 1];
                if (tx < -1 || ty < -1) visible = false;
            }

            // FRI-damped position (Fix F7)
            cur.x += (tx - cur.x) * alpha;
            cur.y += (ty - cur.y) * alpha;

            // Dot instance matrix
            tmpPosition.set(cur.x, cur.y, Z.PULSE);
            tmpScale.set(visible ? 1 : 0, visible ? 1 : 0, 1); // hide by scaling to 0
            tmpMatrix.compose(tmpPosition, tmpQuat, tmpScale);
            dots.setMatrixAt(i, tmpMatrix);

            // Status color
            const status = alertMap[i] || 'green';
            const color = status === 'red' ? COLOR_RED
                : status === 'amber' ? COLOR_AMBER
                    : COLOR_GREEN;
            dots.setColorAt(i, color);
            dotsColorDirty = true;

            // Alert ring
            const showRing = visible && (status === 'red' || status === 'amber');
            if (showRing) {
                const ringScale = status === 'red'
                    ? 1 + Math.sin(t * 4 + i) * 0.3
                    : 1;
                tmpPosition.set(cur.x, cur.y, Z.PULSE - 0.01);
                tmpScale.set(ringScale, ringScale, 1);
                tmpMatrix.compose(tmpPosition, tmpQuat, tmpScale);
                rings.setColorAt(i, color);
            } else {
                // Hide ring by scaling to 0
                tmpPosition.set(0, 0, -10);
                tmpScale.set(0, 0, 1);
                tmpMatrix.compose(tmpPosition, tmpQuat, tmpScale);
            }
            rings.setMatrixAt(i, tmpMatrix);
            ringsColorDirty = true;
        }

        dots.instanceMatrix.needsUpdate = true;
        rings.instanceMatrix.needsUpdate = true;
        if (dotsColorDirty) dots.instanceColor.needsUpdate = true;
        if (ringsColorDirty) rings.instanceColor.needsUpdate = true;
    });

    return (
        <group renderOrder={2}>
            <instancedMesh ref={dotsRef} args={[dotGeo, dotMat, PLAYER_COUNT]} />
            <instancedMesh ref={ringsRef} args={[ringGeo, ringMat, PLAYER_COUNT]} />
        </group>
    );
}
