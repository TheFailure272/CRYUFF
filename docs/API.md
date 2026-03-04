# C.R.U.Y.F.F. — API Reference

## Overview

**C.R.U.Y.F.F.** (Computational Real-time Unified Yield Field Framework) is a live tactical intelligence platform for elite football.

Base URL: `http://192.168.1.10:8000` (stadium edge server)

---

## Endpoints

### WebSocket: Tracking Ingest
**`/ws/tracking`**

25Hz optical tracking data ingest from the Tactical Glass tablet.

| Direction | Format | Description |
|---|---|---|
| Client → Server | JSON `TrackingFrame` | Player positions, ball, timestamps |
| Server → Client | JSON `AnalysisResult` | Tactical overlays, ghost runs, voids |

### WebSocket: Ghost RPC
**`/ws/ghost`**

On-demand Ghost Engine optimization requests.

| Direction | Format | Description |
|---|---|---|
| Client → Server | JSON `{"type": "optimize", ...}` | Request ghost run calculation |
| Server → Client | JSON `GhostResult` | Optimal player positions |

### WebSocket: Voice NLP
**`/ws/voice`**

Push-to-talk voice command endpoint.

| Direction | Format | Description |
|---|---|---|
| Client → Server | Binary (PCM16 250ms chunks) | Audio from RNNoise-filtered mic |
| Client → Server | JSON `{"type": "voice_flush"}` | End of utterance |
| Client → Server | JSON `{"type": "enroll", "audio": [...]}` | Manager voice enrollment (F46) |
| Server → Client | JSON `{"type": "transcript", ...}` | Whisper transcription |
| Server → Client | JSON `{"type": "intent", ...}` | Parsed tactical intent |
| Server → Client | JSON `{"type": "clip", ...}` | Generated clip manifest |

### WebSocket: Omni-Cam
**`/ws/omnicam`**

3DGS Volumetric virtual camera control.

| Direction | Format | Description |
|---|---|---|
| Client → Server | JSON `{"type": "omnicam_pose", "pos": [...], ...}` | Virtual camera pose (Kalman-predicted) |
| Client → Server | JSON `{"type": "ptz_update", "camera_idx": 0, "homography": [...]}` | PTZ extrinsic update (F43) |
| Server → Client | JSON `{"type": "omnicam_ack", "frame": N, "lighting": {...}}` | Acknowledgement + SH state (F47) |

### WebSocket: Dressing Room
**`/ws/dressing-room`**

Apple TV halftime endpoint. LAN-only (F38).

| Direction | Format | Description |
|---|---|---|
| Server → Client | JSON `PushPayload` | Tactical insights + telemetry + HLS |

### POST: Push to Dressing Room
**`POST /api/push`**

Push selected insights to the dressing room Apple TV.

**Request:**
```json
{
  "insight_ids": ["ghost_42", "void_17"],
  "ttl_minutes": 15
}
```

**Response (200):**
```json
{
  "push_id": "push_abc123",
  "insights": 2,
  "telemetry_frames": 375,
  "hls_segments": 8,
  "expires_at": "1709550000.0"
}
```

### POST: WebRTC Offer
**`POST /webrtc/offer`**

WebRTC SDP exchange for unreliable UDP data channel.

### GET: Health Check
**`GET /health`**

```json
{
  "status": "operational",
  "system": "C.R.U.Y.F.F.",
  "webrtc": "available",
  "peers": 2,
  "voice_engine": "ready",
  "volumetric": "ready"
}
```

---

## Deployment

```bash
# Stadium edge server
docker compose up -d

# Development
uvicorn server.main:app --reload --port 8000
```

## Architecture

```
iPad (Tactical Glass)
  ├── /ws/tracking  → Tracking → Analysis → Ghost Engine
  ├── /ws/voice     → Whisper → Intent → Clip Engine
  ├── /ws/omnicam   → Touch Kalman → 3DGS Render
  └── POST /api/push → Push Service → Redis

Apple TV (Dressing Room)
  └── /ws/dressing-room → Telemetry + HLS + Insights
```
