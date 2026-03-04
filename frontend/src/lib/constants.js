/**
 * C.R.U.Y.F.F. — Constants
 *
 * Pitch dimensions, colors, z-layers, and animation config.
 */

// FIFA standard pitch (normalised to [0,1] by backend)
export const PITCH = {
    WIDTH: 1.0,       // x-axis (touchline to touchline)
    HEIGHT: 1.0,      // y-axis (goal line to goal line)
    ASPECT: 105 / 68, // real-world aspect ratio
};

// Strict Z-depth layering (Fix F4 — no Z-fighting)
export const Z = {
    PITCH_LINES: 0.0,
    FLOW: 0.1,
    PULSE: 0.2,
    GHOST: 0.3,
};

// Colors (HSL-tuned for premium dark theme)
export const COLORS = {
    FLOW_BLUE: 0x00b4d8,
    FLOW_CYAN: 0x00f5d4,
    PULSE_GREEN: 0x22c55e,
    PULSE_AMBER: 0xfbbf24,
    PULSE_RED: 0xef4444,
    GHOST_WHITE: 0xf0f0ff,
    PITCH_LINE: 0xffffff,
    PITCH_BG: 0x0a0a14,
};

// Animation
export const ANIM = {
    DAMP_LAMBDA: 12,          // FRI damping: α = 1 - exp(-λ * Δt) (Fix F7)
    BREATHE_SPEED: 1.5,      // void pulse speed (rad/s)
    BREATHE_AMPLITUDE: 0.08, // void scale oscillation (±8%)
    GHOST_DASH_SPEED: 2.0,   // dash offset per second
    GHOST_FADE_SECS: 5.0,    // ghost fades after this many seconds
};

// Transport
export const TRANSPORT = {
    WS_URL: `ws://${window.location.host}/ws/tracking`,
    RECONNECT_BASE_MS: 1000,
    RECONNECT_MAX_MS: 30000,
    HUD_UPDATE_HZ: 4,        // React state updates for HUD (low freq)
};
