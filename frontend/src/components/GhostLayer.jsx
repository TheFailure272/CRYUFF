/**
 * C.R.U.Y.F.F. — GhostLayer (Layer 3: Trajectory Diffusion)
 *
 * Renders the Ghost trajectory as an animated dashed line.
 * - z=0.3 (Fix F4)
 * - Animated dash offset ("running" along the path)
 * - xT badge at the trajectory endpoint
 * - Fades out after GHOST_FADE_SECS
 *
 * Fix F1: Reads ghost data from mutable ref in useFrame.
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Z, COLORS, ANIM } from '../lib/constants';

export default function GhostLayer({ dataRef }) {
    const lineRef = useRef();
    const lastGhostTime = useRef(0);
    const lastGhostId = useRef(null);

    // Pre-allocate a line with max 50 waypoints
    const maxPoints = 50;
    const positions = useMemo(() => new Float32Array(maxPoints * 3), []);
    const geometry = useMemo(() => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setDrawRange(0, 0);
        return geo;
    }, [positions]);

    const material = useMemo(() => new THREE.LineDashedMaterial({
        color: COLORS.GHOST_WHITE,
        transparent: true,
        depthTest: false,
        dashSize: 0.015,
        gapSize: 0.008,
        linewidth: 1,
    }), []);

    useFrame(({ clock }) => {
        const data = dataRef.current;
        const ghost = data?.ghost;
        const line = lineRef.current;
        if (!line) return;

        const now = clock.getElapsedTime();

        // New ghost data arrived
        if (ghost && ghost.waypoints?.length >= 2) {
            const ghostId = JSON.stringify(ghost.waypoints[0]);
            if (ghostId !== lastGhostId.current) {
                lastGhostId.current = ghostId;
                lastGhostTime.current = now;

                // Update positions
                const wp = ghost.waypoints;
                const count = Math.min(wp.length, maxPoints);
                for (let i = 0; i < count; i++) {
                    positions[i * 3] = wp[i].x;
                    positions[i * 3 + 1] = wp[i].y;
                    positions[i * 3 + 2] = Z.GHOST;
                }
                geometry.attributes.position.needsUpdate = true;
                geometry.setDrawRange(0, count);
                geometry.computeBoundingSphere();
                line.computeLineDistances();
            }
        }

        // Fade out after GHOST_FADE_SECS
        const age = now - lastGhostTime.current;
        if (lastGhostTime.current > 0 && age < ANIM.GHOST_FADE_SECS) {
            line.visible = true;
            material.opacity = Math.max(0, 1 - age / ANIM.GHOST_FADE_SECS);

            // Animate dash offset (line "runs" along path)
            material.dashOffset -= ANIM.GHOST_DASH_SPEED * 0.016; // ~60fps
        } else {
            line.visible = false;
        }
    });

    return (
        <group renderOrder={3}>
            <line ref={lineRef} geometry={geometry} material={material} visible={false} />
        </group>
    );
}
