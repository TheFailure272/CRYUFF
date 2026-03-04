/**
 * C.R.U.Y.F.F. — OmniCamLayer (Feature 1)
 *
 * Volumetric "Omni-Cam" — free-viewpoint pitch view via WebRTC.
 *
 * Architecture:
 *   1. WebRTC MediaStreamTrack streams H.264 from server
 *   2. HTML5 <video> element receives the stream
 *   3. Video frame → THREE.VideoTexture → R3F plane
 *   4. Touch events → TouchKalman (80ms prediction) → DataChannel
 *
 * Fix F39: Server renders semantic-masked 3DGS (static stadium +
 *          dynamic player meshes). No cardboard cutout artifacts.
 */

import { useRef, useEffect, useMemo, useCallback } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { TouchKalman } from '../lib/touchKalman';

export default function OmniCamLayer({
    active,
    peerConnection,
    dataChannel,
}) {
    const meshRef = useRef();
    const videoRef = useRef(document.createElement('video'));
    const kalmanRef = useRef(new TouchKalman());
    const textureRef = useRef(null);

    // Setup WebRTC video track
    useEffect(() => {
        if (!active || !peerConnection) return;

        const video = videoRef.current;
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;

        const handleTrack = (event) => {
            if (event.track.kind === 'video') {
                video.srcObject = new MediaStream([event.track]);
                textureRef.current = new THREE.VideoTexture(video);
                textureRef.current.minFilter = THREE.LinearFilter;
                textureRef.current.magFilter = THREE.LinearFilter;
                console.log('[OmniCam] WebRTC video track received');
            }
        };

        peerConnection.addEventListener('track', handleTrack);

        return () => {
            peerConnection.removeEventListener('track', handleTrack);
            if (textureRef.current) {
                textureRef.current.dispose();
                textureRef.current = null;
            }
        };
    }, [active, peerConnection]);

    // Touch → Kalman prediction → DataChannel
    const handlePointerMove = useCallback((event) => {
        if (!dataChannel || dataChannel.readyState !== 'open') return;

        const predicted = kalmanRef.current.update(
            event.point.x,
            event.point.y,
            performance.now(),
        );

        // Send predicted pose (80ms ahead) to server
        dataChannel.send(JSON.stringify({
            type: 'omnicam_pose',
            pos: [predicted.x * 105, predicted.y * 68, 15],
            vx: predicted.vx,
            vy: predicted.vy,
        }));
    }, [dataChannel]);

    const handlePointerDown = useCallback(() => {
        kalmanRef.current.reset();
    }, []);

    // Geometry for fullscreen video plane
    const planeGeo = useMemo(() =>
        new THREE.PlaneGeometry(1, 1), []);

    const planeMat = useMemo(() =>
        new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            depthTest: false,
        }), []);

    useFrame(() => {
        if (!meshRef.current || !active) return;

        if (textureRef.current) {
            meshRef.current.material.map = textureRef.current;
            meshRef.current.material.needsUpdate = true;
        }

        meshRef.current.visible = active;
    });

    if (!active) return null;

    return (
        <mesh
            ref={meshRef}
            geometry={planeGeo}
            material={planeMat}
            position={[0.5, 0.5, -0.01]}
            onPointerMove={handlePointerMove}
            onPointerDown={handlePointerDown}
        />
    );
}
