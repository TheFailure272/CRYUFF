/**
 * C.R.U.Y.F.F. — VoiceButton (Feature 2)
 *
 * Push-to-talk button with streaming audio via WebRTC DataChannel.
 *
 * Fix F36: RNNoise noise suppression applied BEFORE streaming.
 * Audio flows: Mic → AudioWorklet (RNNoise) → MediaRecorder (250ms) → DataChannel
 *
 * The manager holds the button to speak. Audio streams live.
 * On release, a final flush triggers Whisper inference.
 */

import { useRef, useState, useCallback } from 'react';

const CHUNK_INTERVAL = 250; // ms — 250ms audio chunks

/**
 * Attempts to apply RNNoise-based noise suppression.
 * Falls back to browser's built-in noise suppression if unavailable.
 */
async function getCleanStream() {
    const constraints = {
        audio: {
            channelCount: 1,
            sampleRate: 16000,
            // Fix F36: Request browser-level noise suppression
            // This is the fallback if WebRTC RNNoise worklet isn't available
            noiseSuppression: true,
            echoCancellation: true,
            autoGainControl: true,
        },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    // Attempt to load RNNoise AudioWorklet for superior suppression
    // (defeats 110dB stadium roar far better than browser defaults)
    try {
        const ctx = new AudioContext({ sampleRate: 16000 });
        await ctx.audioWorklet.addModule('/audio/rnnoise-worklet.js');
        const source = ctx.createMediaStreamSource(stream);
        const rnnoise = new AudioWorkletNode(ctx, 'rnnoise-processor');
        const dest = ctx.createMediaStreamDestination();
        source.connect(rnnoise);
        rnnoise.connect(dest);
        console.log('[VoiceButton] RNNoise worklet loaded — stadium-grade suppression active');
        return dest.stream;
    } catch {
        console.warn('[VoiceButton] RNNoise unavailable — using browser noise suppression');
        return stream;
    }
}

export default function VoiceButton({ dataChannel, onTranscript, onStatus }) {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState('');
    const recorderRef = useRef(null);
    const streamRef = useRef(null);

    const startRecording = useCallback(async () => {
        if (!dataChannel || dataChannel.readyState !== 'open') {
            onStatus?.('DataChannel not open');
            return;
        }

        try {
            // Get noise-suppressed audio stream (Fix F36)
            const stream = await getCleanStream();
            streamRef.current = stream;

            const recorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
            });

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0 && dataChannel.readyState === 'open') {
                    // Stream chunk to backend via DataChannel
                    e.data.arrayBuffer().then((buf) => {
                        dataChannel.send(buf);
                    });
                }
            };

            // Emit chunks every 250ms
            recorder.start(CHUNK_INTERVAL);
            recorderRef.current = recorder;

            setIsRecording(true);
            setTranscript('');
            onStatus?.('listening');
        } catch (err) {
            console.error('[VoiceButton] Mic access failed:', err);
            onStatus?.('mic_error');
        }
    }, [dataChannel, onStatus]);

    const stopRecording = useCallback(() => {
        if (recorderRef.current && recorderRef.current.state !== 'inactive') {
            recorderRef.current.stop();
        }

        // Stop all tracks
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }

        // Signal flush to backend (button release)
        if (dataChannel && dataChannel.readyState === 'open') {
            dataChannel.send(JSON.stringify({ type: 'voice_flush' }));
        }

        setIsRecording(false);
        onStatus?.('processing');
    }, [dataChannel, onStatus]);

    return (
        <div className="voice-container">
            <button
                className={`voice-button ${isRecording ? 'recording' : ''}`}
                onPointerDown={startRecording}
                onPointerUp={stopRecording}
                onPointerLeave={stopRecording}
                id="voice-ptt-button"
            >
                <span className="voice-icon">
                    {isRecording ? '🔴' : '🎙️'}
                </span>
                <span className="voice-label">
                    {isRecording ? 'Listening…' : 'Voice'}
                </span>
            </button>

            {transcript && (
                <div className="voice-transcript fade-in">
                    {transcript}
                </div>
            )}
        </div>
    );
}
