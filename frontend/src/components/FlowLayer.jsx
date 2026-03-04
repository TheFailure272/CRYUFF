/**
 * C.R.U.Y.F.F. — FlowLayer (Layer 1: Topology Voids) — Final
 *
 * Fixes applied:
 * - Fix F7:  Frame-rate independent damping via 1 - exp(-λ * Δt)
 * - Fix F12: Custom ShaderMaterial with explicit `precision highp float`
 *            to prevent gradient banding on mobile GPUs (mediump default).
 *
 * The custom shader renders each void as a radial-gradient disc:
 * - Center: bright cyan/blue, fading to transparent at edges
 * - Produces a premium soft-glow halo effect
 * - highp guarantees smooth gradients on all tablet hardware
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Z, COLORS, ANIM } from '../lib/constants';

const MAX_VOIDS = 8;

const colorBlue = new THREE.Color(COLORS.FLOW_BLUE);
const colorCyan = new THREE.Color(COLORS.FLOW_CYAN);
const tmpColor = new THREE.Color();

/** Frame-rate independent damping factor */
function friAlpha(delta) {
    return 1 - Math.exp(-ANIM.DAMP_LAMBDA * delta);
}

/**
 * Fix F12: Custom ShaderMaterial with forced highp precision.
 *
 * - Vertex shader: standard transform (model-view-projection)
 * - Fragment shader: radial gradient from center (opaque) to edge (transparent)
 *   with smooth falloff. Forces highp to prevent banding on mobile GPUs.
 */
function createGlowMaterial() {
    return new THREE.ShaderMaterial({
        transparent: true,
        depthTest: false,
        side: THREE.DoubleSide,
        uniforms: {
            uColor: { value: new THREE.Color(COLORS.FLOW_BLUE) },
            uOpacity: { value: 0.0 },
        },
        vertexShader: /* glsl */ `
      precision highp float;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
        fragmentShader: /* glsl */ `
      precision highp float;
      uniform vec3 uColor;
      uniform float uOpacity;
      varying vec2 vUv;

      void main() {
        // Distance from center (0,0) to edge (0.5) — UV is [0,1]
        vec2 center = vUv - 0.5;
        float dist = length(center) * 2.0;  // normalise to [0,1]

        // Smooth radial falloff with cubic ease
        float glow = 1.0 - smoothstep(0.0, 1.0, dist);
        glow = glow * glow;  // quadratic falloff for softer glow

        // Final color with premultiplied alpha
        float alpha = glow * uOpacity;
        gl_FragColor = vec4(uColor * alpha, alpha);
      }
    `,
    });
}

export default function FlowLayer({ dataRef, degradation }) {
    const meshRefs = useRef([]);
    const targets = useRef(
        Array.from({ length: MAX_VOIDS }, () => ({
            x: 0, y: 0, scale: 0, opacity: 0, persistence: 0,
        }))
    );

    const geometry = useMemo(() => new THREE.CircleGeometry(1, 32), []);

    // Pre-create materials (one per void for independent colors)
    const materials = useMemo(
        () => Array.from({ length: MAX_VOIDS }, () => createGlowMaterial()),
        []
    );

    useFrame(({ clock }, delta) => {
        const data = dataRef.current;
        const voids = data?.voids || [];
        const t = clock.getElapsedTime();
        const alpha = friAlpha(delta);

        // Update targets from latest data
        for (let i = 0; i < MAX_VOIDS; i++) {
            const target = targets.current[i];
            if (i < voids.length) {
                const v = voids[i];
                target.x = v.centroid_x;
                target.y = v.centroid_y;
                target.scale = Math.max(0.02, v.persistence * 0.6);
                target.opacity = Math.min(1, v.stability || 0);
                target.persistence = v.persistence;
            } else {
                target.opacity = 0;
                target.scale = 0;
            }
        }

        // Apply to meshes with FRI damping
        for (let i = 0; i < MAX_VOIDS; i++) {
            const mesh = meshRefs.current[i];
            if (!mesh) continue;
            const target = targets.current[i];
            const mat = materials[i];

            // FRI-damped position
            mesh.position.x += (target.x - mesh.position.x) * alpha;
            mesh.position.y += (target.y - mesh.position.y) * alpha;
            mesh.position.z = Z.FLOW;

            // Breathing animation (Fix F16: disabled on low battery)
            const breatheEnabled = degradation?.enableBreathing !== false;
            const breathe = breatheEnabled
                ? 1 + Math.sin(t * ANIM.BREATHE_SPEED + i) * ANIM.BREATHE_AMPLITUDE
                : 1;
            const s = target.scale * breathe;
            mesh.scale.set(s, s, 1);

            // Opacity FRI damp (uses shader uniform)
            const targetOpacity = target.opacity * 0.6;
            mat.uniforms.uOpacity.value +=
                (targetOpacity - mat.uniforms.uOpacity.value) * alpha;

            // Color: blue → cyan by persistence
            const ratio = Math.min(1, target.persistence / 0.2);
            tmpColor.copy(colorBlue).lerp(colorCyan, ratio);
            mat.uniforms.uColor.value.copy(tmpColor);

            mesh.visible = target.opacity > 0.01;
        }
    });

    return (
        <group renderOrder={1}>
            {Array.from({ length: MAX_VOIDS }, (_, i) => (
                <mesh
                    key={i}
                    ref={(el) => { meshRefs.current[i] = el; }}
                    geometry={geometry}
                    material={materials[i]}
                    visible={false}
                />
            ))}
        </group>
    );
}
