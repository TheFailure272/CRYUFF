/**
 * C.R.U.Y.F.F. — PitchLines
 *
 * Football pitch markings as thin white lines on a plane.
 * z=0 (Fix F4), opacity 0.15 to not compete with data layers.
 *
 * Coordinates are in normalised [0, 1] × [0, 1] space.
 * Real pitch: 105m × 68m.
 */

import { useMemo } from 'react';
import * as THREE from 'three';
import { Z, COLORS } from '../lib/constants';

// Pitch geometry helpers (all in normalised 0-1 space)
// x: 0 = left touchline, 1 = right touchline
// y: 0 = bottom goal line, 1 = top goal line

function createPitchGeometry() {
    const points = [];

    const line = (x1, y1, x2, y2) => {
        points.push(new THREE.Vector3(x1, y1, Z.PITCH_LINES));
        points.push(new THREE.Vector3(x2, y2, Z.PITCH_LINES));
    };

    // Touchlines & goal lines (outer boundary)
    line(0, 0, 1, 0); // bottom
    line(1, 0, 1, 1); // right
    line(1, 1, 0, 1); // top
    line(0, 1, 0, 0); // left

    // Halfway line
    line(0, 0.5, 1, 0.5);

    // Penalty areas (16.5m from goal line = 16.5/68 ≈ 0.2426)
    // Width: 40.3m centered = (105-40.3)/2/105 ≈ 0.308 from each side
    const paY = 16.5 / 68;
    const paXl = (105 - 40.3) / 2 / 105;
    const paXr = 1 - paXl;

    // Bottom penalty area
    line(paXl, 0, paXl, paY);
    line(paXl, paY, paXr, paY);
    line(paXr, paY, paXr, 0);

    // Top penalty area
    line(paXl, 1, paXl, 1 - paY);
    line(paXl, 1 - paY, paXr, 1 - paY);
    line(paXr, 1 - paY, paXr, 1);

    // Goal areas (5.5m from goal line, 18.3m wide centered)
    const gaY = 5.5 / 68;
    const gaXl = (105 - 18.3) / 2 / 105;
    const gaXr = 1 - gaXl;

    // Bottom goal area
    line(gaXl, 0, gaXl, gaY);
    line(gaXl, gaY, gaXr, gaY);
    line(gaXr, gaY, gaXr, 0);

    // Top goal area
    line(gaXl, 1, gaXl, 1 - gaY);
    line(gaXl, 1 - gaY, gaXr, 1 - gaY);
    line(gaXr, 1 - gaY, gaXr, 1);

    // Penalty spots (11m from goal = 11/68 ≈ 0.1618)
    const psY = 11 / 68;
    const psR = 0.003;
    // Bottom penalty spot (small cross)
    line(0.5 - psR, psY, 0.5 + psR, psY);
    line(0.5, psY - psR, 0.5, psY + psR);
    // Top penalty spot
    line(0.5 - psR, 1 - psY, 0.5 + psR, 1 - psY);
    line(0.5, 1 - psY - psR, 0.5, 1 - psY + psR);

    return points;
}

function createCirclePoints(cx, cy, r, segments = 48) {
    const pts = [];
    for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        pts.push(new THREE.Vector3(
            cx + Math.cos(a) * r,
            cy + Math.sin(a) * r,
            Z.PITCH_LINES,
        ));
    }
    return pts;
}

export default function PitchLines() {
    const { lineSegments, circleGeo } = useMemo(() => {
        const pts = createPitchGeometry();
        const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);

        // Centre circle (9.15m radius = 9.15/105 ≈ 0.0871 in x-normalised)
        const crPts = createCirclePoints(0.5, 0.5, 9.15 / 105, 64);
        const crGeo = new THREE.BufferGeometry().setFromPoints(crPts);

        return { lineSegments: lineGeo, circleGeo: crGeo };
    }, []);

    const mat = useMemo(() => new THREE.LineBasicMaterial({
        color: COLORS.PITCH_LINE,
        transparent: true,
        opacity: 0.15,
        depthTest: false,
    }), []);

    return (
        <group renderOrder={0}>
            <lineSegments geometry={lineSegments} material={mat} />
            <line geometry={circleGeo} material={mat} />
        </group>
    );
}
