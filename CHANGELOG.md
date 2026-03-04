# Changelog

All notable changes to C.R.U.Y.F.F. are documented in this file.

## [1.0.0] — 2026-03-04

### 🏗️ Core Architecture (F1–F15)

- **F1** — FastAPI WebSocket gateway with JSON frame validation
- **F2** — Redis pub/sub for zero-copy frame distribution
- **F3** — Pydantic `TrackingFrame` schema with 44-float coordinate validation
- **F4** — Worker process offloading analysis from the event loop
- **F5** — Shared memory buffer (`SHMBuffer`) for inter-process data exchange
- **F6** — Client-side 30-second telemetry ring buffer (constant memory, O(1) ops)
- **F7** — WebSocket binary frame encoding for bandwidth reduction
- **F8** — Thread pool executor for CPU-bound topology computation
- **F9** — Connection Manager with broadcast and client lifecycle management
- **F10** — Pydantic Settings for environment-driven configuration
- **F11** — Monotonic timestamp tagging, independent of NTP drift
- **F12** — Stale frame detection and eviction (>100ms → drop)
- **F13** — `ingest_monotonic` tagging on every incoming tracking frame
- **F14** — Redis ZRANGEBYSCORE for temporal telemetry queries
- **F15** — Health check endpoint with WebRTC + peer status

### 🌐 Network Hardening (F16–F22)

- **F16** — Reconnection rate limiter with exponential backoff (thundering herd prevention)
- **F17** — WebRTC DataChannel (UDP, unreliable, unordered) for analysis results
- **F18** — SDP offer/answer exchange via `POST /webrtc/offer`
- **F19** — Transport hysteresis: WebRTC → WebSocket → reconnect (no flapping)
- **F20** — Graceful degradation: reduces rendering quality on frame drop
- **F21** — Context recovery hook: restores state after reconnection
- **F22** — Video sync hook: compensates for broadcast→tracking timestamp jitter

### 📐 Topology & Visualization (F23–F32)

- **F23** — Vietoris–Rips complex on defensive point cloud via GUDHI
- **F24** — β₁ persistent homology for topological void (passing lane) detection
- **F25** — Death-triangle centroid heuristic for void localization
- **F26** — Dynamic team partitioning (team_ids-first, fallback to index split)
- **F27** — Kinematic velocity projection (topology of the *future* defensive shape)
- **F28** — Kalman-filtered void tracking with Hungarian Algorithm matching
- **F29** — Spatial Boundary Clamp: constrains coordinates to [0, 1] pitch space
- **F30** — Stability threshold: only emit voids with stability ≥ 0.6
- **F31** — R3F (React Three Fiber) GPU-accelerated pitch rendering
- **F32** — HUD overlay: match timer, score, connection status, mode selector

### 🫀 Wearable Fusion (F33)

- **F33** — Joseph-form Extended Kalman Filter: 7-state [x, y, vx, vy, ax, ay, hr]. Guarantees positive semi-definite covariance over 90+ min continuous runtime. Mahalanobis gate at χ²(2, 0.99) = 9.21 rejects GPS multipath while preserving heart rate

### ⚽ Dead-Ball Solver (F34–F35)

- **F34** — Biomechanical Jump Window: headers constrained to 0.3–0.7s flight time, 0.65m max jump height
- **F35** — Kinematic foul penalty: rejects ghost collisions at >15 km/h closing speed

### 🎙️ Voice NLP — Initial (F36–F37)

- **F36** — Rolling VAD buffer with 5-second Max-Token Guillotine (prevents RAM overflow in stadium roar). RNNoise AudioWorklet for pre-streaming noise suppression
- **F37** — Packaged telemetry: each historical clip carries its own 25Hz frame block, bypassing the live ring buffer

### 📺 Dressing Room Push (F38)

- **F38** — Air-gapped Ethernet LAN push. All HLS URLs on 192.168.x.x intranet. GOP=1 segments for instant-start playback. Late-joiner catch-up. Survives the concrete Faraday cage of stadium bowels

### 🎥 Volumetric Omni-Cam — Initial (F39)

- **F39** — SAM semantic masking: players separated from static stadium before 3DGS rendering. Eliminates "cardboard cutout" artifacts at novel viewpoints

### 🎙️ Voice NLP — Hardening (F40)

- **F40** — Hot-word prompting: injects tactical lexicon ("half-space", "double pivot", "Cruyff turn", "salida lavolpiana") as Whisper's `initial_prompt`, biasing beam search toward correct jargon

### ⚽ Dead-Ball Solver — Physics (F41)

- **F41** — Reynolds-dependent drag crisis: Cd = 0.25 (subcritical, Re < 2×10⁵) → 0.15 (turbulent, Re > 4×10⁵). Smooth `jnp.clip` interpolation. Models the knuckleball effect — ball carries at speed, dips on deceleration

### 🫀 Biomechanics (F42)

- **F42** — Scapular offset: GPS pod between shoulder blades shifts 0.5–1.0m during sprint. Offset scaled by acceleration, projected backward along velocity vector

### 🎥 Volumetric Omni-Cam — PTZ (F43)

- **F43** — PTZ extrinsic synchronization: ingests 25Hz homography matrices from broadcast cameras, dynamically updates 3DGS camera extrinsics. Prevents geometric shearing when cameras pan/tilt/zoom

### 🎙️ Voice NLP — Match Clock (F44)

- **F44** — VAR Match-Time Paradox: resolves "10 minutes ago" against the official OPTA match clock (Redis `cruyff:match_clock`), not wall-clock time. Accounts for VAR stoppages and injury time

### 🫀 Biomechanics — Anatomical (F45)

- **F45** — Anatomical Posterior Axis: replaces velocity-based scapular offset with optical hip facing angle. Correctly handles backpedaling defenders (velocity points backward, but offset must project behind the player's *body*, not their direction of travel)

### 🎙️ Voice NLP — Diarization (F46)

- **F46** — Zero-shot speaker diarization: pre-enrolled manager voiceprint (Resemblyzer). 500ms audio windows compared via cosine similarity (threshold 0.75). Cross-talk from assistant coach / fitness coach is muted before Whisper inference

### 🎥 Volumetric Omni-Cam — Lighting (F47)

- **F47** — Dynamic Spherical Harmonic modulation: every 60s, extracts BT.709 luminance and R/B colour temperature from live broadcast. Globally shifts the 0th-order SH coefficient (Y₀⁰) of background splats. Prevents sunset→floodlight visual mismatch

### 🔧 JAX Compatibility (F48)

- **F48** — JAX traceability: replaced Python `if/elif` in `_compute_cd_reynolds` with `jnp.clip`, Python `for`/`if` in `_simulate_single` with `jax.lax.fori_loop` + `jnp.where`. Removed `float()` casts. 100 trajectories: 428ms (JAX CPU) vs timeout (NumPy)

### 🚀 Deployment Pipeline

- Server routes: `/ws/voice`, `POST /api/push`, `/ws/dressing-room`, `/ws/omnicam`
- Integration tests: 76 passed in 2.39s
- Docker Compose: Redis + backend + frontend
- CI/CD: GitHub Actions (lint → test → build → docker)
- Performance profiling script
- API reference documentation (`docs/API.md`)
- Comprehensive README with 90-minute match scenario
