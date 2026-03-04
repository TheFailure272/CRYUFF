/**
 * C.R.U.Y.F.F. — SetPieceOverlay (Feature 4)
 *
 * Activated when the backend detects a dead-ball situation.
 * Renders:
 *   1. Heatmap of ball landing probability (red=high, blue=low)
 *   2. Golden delivery zone marker (highest combined score)
 *   3. Ghost runner paths (optimal attacking runs)
 *   4. Blocking screen positions
 *
 * Data arrives in dataRef.current.setpiece when active.
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Z } from '../lib/constants';

const HEATMAP_Z = Z.FLOW + 0.05;
const ZONE_Z = Z.PULSE + 0.05;
const GHOST_Z = Z.GHOST + 0.05;

// Pitch normalisation (meters → [0,1])
const PITCH_W = 105;
const PITCH_H = 68;

const COLOR_HOT = new THREE.Color(0xef4444);    // red — high probability
const COLOR_COLD = new THREE.Color(0x1e3a5f);   // dark blue — low
const COLOR_GOLD = new THREE.Color(0xffd700);    // golden delivery zone
const COLOR_RUN = new THREE.Color(0x22c55e);     // green — attacker runs
const COLOR_SCREEN = new THREE.Color(0xfbbf24);  // amber — blocking screen

const tmpColor = new THREE.Color();

export default function SetPieceOverlay({ dataRef }) {
    const heatmapRef = useRef();
    const zoneRef = useRef();
    const runsRef = useRef();
    const screenRef = useRef();

    // Heatmap grid — pre-allocate max 40×17 cells
    const MAX_CELLS = 40 * 17;
    const cellGeo = useMemo(() =>
        new THREE.PlaneGeometry(1 / PITCH_W, 1 / PITCH_H), []);
    const cellMat = useMemo(() => new THREE.MeshBasicMaterial({
        transparent: true,
        depthTest: false,
        opacity: 0.5,
    }), []);

    // Zone marker
    const zoneGeo = useMemo(() =>
        new THREE.RingGeometry(0.015, 0.025, 24), []);
    const zoneMat = useMemo(() => new THREE.MeshBasicMaterial({
        color: COLOR_GOLD,
        transparent: true,
        depthTest: false,
        opacity: 0.9,
    }), []);

    // Run paths (instanced lines as thin quads)
    const MAX_RUNS = 9;  // 3 zones × 3 runs
    const runGeo = useMemo(() =>
        new THREE.PlaneGeometry(0.003, 1), []);  // thin line
    const runMat = useMemo(() => new THREE.MeshBasicMaterial({
        color: COLOR_RUN,
        transparent: true,
        depthTest: false,
        opacity: 0.7,
    }), []);

    // Blocking screen dots
    const MAX_SCREENS = 9;
    const screenGeo = useMemo(() =>
        new THREE.CircleGeometry(0.008, 8), []);
    const screenMat = useMemo(() => new THREE.MeshBasicMaterial({
        color: COLOR_SCREEN,
        transparent: true,
        depthTest: false,
        opacity: 0.8,
    }), []);

    const tmpMatrix = useMemo(() => new THREE.Matrix4(), []);

    useFrame(({ clock }) => {
        const data = dataRef.current;
        const sp = data?.setpiece;

        const heatmapMesh = heatmapRef.current;
        const zoneMesh = zoneRef.current;
        const runsMesh = runsRef.current;
        const screenMesh = screenRef.current;

        if (!sp || !heatmapMesh) {
            // No set-piece active — hide everything
            if (heatmapMesh) heatmapMesh.visible = false;
            if (zoneMesh) zoneMesh.visible = false;
            if (runsMesh) runsMesh.visible = false;
            if (screenMesh) screenMesh.visible = false;
            return;
        }

        // ── Heatmap ─────────────────────────────────────
        const heatmap = sp.heatmap;
        const gridX = sp.grid_x;
        const gridY = sp.grid_y;

        if (heatmap && gridX && gridY) {
            heatmapMesh.visible = true;
            const maxVal = Math.max(...heatmap.flat(), 0.001);
            let cellIdx = 0;

            for (let i = 0; i < gridX.length && cellIdx < MAX_CELLS; i++) {
                for (let j = 0; j < gridY.length && cellIdx < MAX_CELLS; j++) {
                    const val = heatmap[i]?.[j] || 0;
                    const nx = gridX[i] / PITCH_W;
                    const ny = gridY[j] / PITCH_H;

                    tmpMatrix.makeTranslation(nx, ny, HEATMAP_Z);
                    heatmapMesh.setMatrixAt(cellIdx, tmpMatrix);

                    // Color interpolation: cold→hot based on probability
                    const t = val / maxVal;
                    tmpColor.lerpColors(COLOR_COLD, COLOR_HOT, t);
                    heatmapMesh.setColorAt(cellIdx, tmpColor);

                    cellIdx++;
                }
            }

            // Hide unused instances
            for (let k = cellIdx; k < MAX_CELLS; k++) {
                tmpMatrix.makeScale(0, 0, 0);
                heatmapMesh.setMatrixAt(k, tmpMatrix);
            }

            heatmapMesh.instanceMatrix.needsUpdate = true;
            if (heatmapMesh.instanceColor)
                heatmapMesh.instanceColor.needsUpdate = true;
        }

        // ── Golden Delivery Zone ────────────────────────
        const topZones = sp.top_zones;
        if (topZones && topZones.length > 0 && zoneMesh) {
            const best = topZones[0];
            const zx = best.x / PITCH_W;
            const zy = best.y / PITCH_H;
            zoneMesh.position.set(zx, zy, ZONE_Z);
            zoneMesh.visible = true;

            // Pulse animation
            const pulse = 1 + Math.sin(clock.getElapsedTime() * 4) * 0.15;
            zoneMesh.scale.set(pulse, pulse, 1);
        }

        // ── Ghost Runner Paths ──────────────────────────
        const plans = sp.plans;
        if (plans && runsMesh) {
            runsMesh.visible = true;
            let runIdx = 0;

            for (const plan of plans) {
                for (const run of (plan.runs || [])) {
                    if (runIdx >= MAX_RUNS) break;

                    const sx = run.start.x / PITCH_W;
                    const sy = run.start.y / PITCH_H;
                    const tx = run.target.x / PITCH_W;
                    const ty = run.target.y / PITCH_H;

                    const mx = (sx + tx) / 2;
                    const my = (sy + ty) / 2;
                    const dx = tx - sx;
                    const dy = ty - sy;
                    const len = Math.sqrt(dx * dx + dy * dy);
                    const angle = Math.atan2(dy, dx);

                    tmpMatrix.makeTranslation(mx, my, GHOST_Z);
                    tmpMatrix.multiply(
                        new THREE.Matrix4().makeRotationZ(angle)
                    );
                    tmpMatrix.multiply(
                        new THREE.Matrix4().makeScale(1, len, 1)
                    );
                    runsMesh.setMatrixAt(runIdx, tmpMatrix);

                    // Color by header probability
                    const hpColor = run.header_prob > 0.5 ? 0x22c55e : 0xfbbf24;
                    tmpColor.set(hpColor);
                    runsMesh.setColorAt(runIdx, tmpColor);

                    runIdx++;
                }
            }

            // Hide unused
            for (let k = runIdx; k < MAX_RUNS; k++) {
                tmpMatrix.makeScale(0, 0, 0);
                runsMesh.setMatrixAt(k, tmpMatrix);
            }

            runsMesh.instanceMatrix.needsUpdate = true;
            if (runsMesh.instanceColor)
                runsMesh.instanceColor.needsUpdate = true;
        }

        // ── Blocking Screen Dots ────────────────────────
        if (plans && screenMesh) {
            screenMesh.visible = true;
            let sIdx = 0;

            for (const plan of plans) {
                for (const s of (plan.blocking_screen || [])) {
                    if (sIdx >= MAX_SCREENS) break;
                    const sx = s.x / PITCH_W;
                    const sy = s.y / PITCH_H;
                    tmpMatrix.makeTranslation(sx, sy, GHOST_Z);
                    screenMesh.setMatrixAt(sIdx, tmpMatrix);
                    sIdx++;
                }
            }

            for (let k = sIdx; k < MAX_SCREENS; k++) {
                tmpMatrix.makeScale(0, 0, 0);
                screenMesh.setMatrixAt(k, tmpMatrix);
            }

            screenMesh.instanceMatrix.needsUpdate = true;
        }
    });

    return (
        <group renderOrder={3}>
            {/* Heatmap — instanced grid cells */}
            <instancedMesh
                ref={heatmapRef}
                args={[cellGeo, cellMat, MAX_CELLS]}
                visible={false}
            />

            {/* Golden delivery zone marker */}
            <mesh ref={zoneRef} geometry={zoneGeo} material={zoneMat}
                visible={false} />

            {/* Ghost runner paths */}
            <instancedMesh
                ref={runsRef}
                args={[runGeo, runMat, MAX_RUNS]}
                visible={false}
            />

            {/* Blocking screen positions */}
            <instancedMesh
                ref={screenRef}
                args={[screenGeo, screenMat, MAX_SCREENS]}
                visible={false}
            />
        </group>
    );
}
