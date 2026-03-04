/**
 * C.R.U.Y.F.F. — ClipDrawer (Feature 2)
 *
 * Playlist drawer for voice-requested historical clips.
 *
 * Fix F37: Isolated Canvas with Packaged Telemetry.
 * Each clip carries its own telemetry JSON block (extracted from Redis).
 * The AR overlay is rendered from this static data, completely
 * bypassing the live 30-second TelemetryRingBuffer.
 *
 * Architecture:
 * - Clip list (left rail, tap to select)
 * - Video player with AR overlay canvas (main view)
 * - The <Canvas> reads from clip.telemetry, NOT useAnalysisStream
 */

import { useState, useRef, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import PitchLines from './PitchLines';
import FlowLayer from './FlowLayer';
import PulseLayer from './PulseLayer';

export default function ClipDrawer({ clips, onClose }) {
    const [selectedIdx, setSelectedIdx] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const videoRef = useRef(null);

    const selectedClip = clips?.[selectedIdx];

    // Fix F37: Create a synthetic dataRef from packaged telemetry
    // This completely bypasses the live useAnalysisStream
    const clipDataRef = useRef({ current: {} });

    useEffect(() => {
        if (!selectedClip || !isPlaying) return;

        const telemetry = selectedClip.telemetry || [];
        if (!telemetry.length) return;

        let frameIdx = 0;
        const fps = 25;
        const interval = setInterval(() => {
            if (frameIdx < telemetry.length) {
                clipDataRef.current = telemetry[frameIdx];
                frameIdx++;
            } else {
                clearInterval(interval);
                setIsPlaying(false);
            }
        }, 1000 / fps);

        return () => clearInterval(interval);
    }, [selectedClip, isPlaying]);

    const handlePlay = () => {
        if (videoRef.current) {
            videoRef.current.play();
            setIsPlaying(true);
        }
    };

    if (!clips || clips.length === 0) {
        return null;
    }

    return (
        <div className="clip-drawer" id="clip-drawer">
            {/* Header */}
            <div className="clip-drawer-header">
                <span className="clip-drawer-title">
                    📹 Tactical Clips ({clips.length})
                </span>
                <button
                    className="clip-drawer-close"
                    onClick={onClose}
                    id="clip-drawer-close"
                >
                    ✕
                </button>
            </div>

            <div className="clip-drawer-body">
                {/* Clip list (left rail) */}
                <div className="clip-list">
                    {clips.map((clip, i) => (
                        <button
                            key={clip.clip_id}
                            className={`clip-item ${i === selectedIdx ? 'active' : ''}`}
                            onClick={() => { setSelectedIdx(i); setIsPlaying(false); }}
                            id={`clip-item-${i}`}
                        >
                            <span className="clip-event">{clip.event_type}</span>
                            <span className="clip-time">
                                {Math.floor(clip.start_s / 60)}′
                                {Math.floor(clip.start_s % 60).toString().padStart(2, '0')}″
                            </span>
                            <span className="clip-frames">
                                {clip.telemetry_frames} frames
                            </span>
                        </button>
                    ))}
                </div>

                {/* Main view: Video + AR overlay */}
                <div className="clip-viewer">
                    {selectedClip && (
                        <>
                            {/* Video layer */}
                            <video
                                ref={videoRef}
                                className="clip-video"
                                src={selectedClip.hls_segments?.[0]}
                                onEnded={() => setIsPlaying(false)}
                            />

                            {/* Fix F37: Isolated AR Canvas
                                Reads from clipDataRef (packaged static JSON),
                                NOT from the live stream */}
                            <div className="clip-ar-overlay">
                                <Canvas
                                    gl={{ alpha: true, antialias: false }}
                                    orthographic
                                    camera={{ zoom: 1 }}
                                    style={{
                                        position: 'absolute',
                                        top: 0, left: 0,
                                        width: '100%', height: '100%',
                                        pointerEvents: 'none',
                                    }}
                                >
                                    <PitchLines />
                                    <FlowLayer dataRef={clipDataRef} />
                                    <PulseLayer dataRef={clipDataRef} />
                                </Canvas>
                            </div>

                            {/* Play controls */}
                            <div className="clip-controls">
                                <button
                                    className="clip-play-btn"
                                    onClick={handlePlay}
                                    disabled={isPlaying}
                                    id="clip-play-button"
                                >
                                    {isPlaying ? '⏸ Playing' : '▶ Play with AR'}
                                </button>
                                {selectedClip.annotation && (
                                    <span className="clip-annotation">
                                        {selectedClip.annotation}
                                    </span>
                                )}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
