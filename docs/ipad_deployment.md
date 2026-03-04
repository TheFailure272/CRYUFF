# C.R.U.Y.F.F. — iPad Deployment Runbook (Fix F25)

## Guided Access: Hardware-Level Gesture Lockdown

> [!CAUTION]
> CSS `touch-action: none` and PWA standalone mode **cannot** override core iPadOS system gestures (Home Indicator swipe, Notification Center pull-down, Control Center). A stressed manager swiping aggressively **will** accidentally background the app. This must be solved at the hardware/MDM level.

## Pre-Match Setup (15 minutes before kickoff)

### Option A: Manual Guided Access (Single Tablet)

1. **Open** the Tactical Glass PWA on Safari
2. **Triple-click** the Side Button → "Guided Access" dialog appears
3. **Disable** all hardware buttons:
   - Volume Buttons: OFF
   - Sleep/Wake Button: OFF
   - Motion: OFF
   - Touch: ON (keep touch enabled)
4. **Set** a 4-digit passcode (shared with kit manager only)
5. **Tap** "Start" → the iPad is now locked to the Tactical Glass

### Option B: Apple Configurator MDM (Fleet Deployment)

For clubs deploying multiple tablets (manager + assistant + analyst):

```bash
# 1. Enroll tablets in Apple Business Manager
# 2. Push MDM profile via Apple Configurator 2:

# Profile payload:
{
  "PayloadType": "com.apple.app.lock",
  "PayloadVersion": 1,
  "App": {
    "Identifier": "com.cruyff.tactical-glass",
    "Options": {
      "DisableTouch": false,
      "DisableDeviceRotation": true,
      "DisableVolumeButtons": true,
      "DisableSleepWakeButton": false,
      "DisableAutoLock": true,
      "EnableVoiceOver": false
    }
  }
}
```

### Option C: Autonomous Single App Mode (ASAM)

For enterprise-enrolled iPads:

1. **Register** the Tactical Glass app for ASAM in the MDM server
2. The app programmatically enters Single App Mode on launch:
   ```swift
   UIAccessibility.requestGuidedAccessSession(enabled: true)
   ```
3. The Home Indicator, Notification Center, and Control Center are all disabled at the OS level
4. Only the MDM admin can unlock the tablet

## Display Settings

| Setting | Value | Reason |
|---|---|---|
| Auto-Lock | Never | Prevents screen sleep during half-time |
| Brightness | Manual, 80% | Prevents adaptive brightness in sunlight |
| True Tone | OFF | Prevents color shift under stadium lights |
| Night Shift | OFF | Preserves blue/cyan tactical color accuracy |
| Text Size | Default | Prevents layout breakage |

## Network Checklist

- [ ] Stadium Wi-Fi SSID pre-configured (WPA3 Enterprise)
- [ ] TURN server endpoint reachable via port 443
- [ ] JWT pre-loaded in localStorage via team IT portal
- [ ] Cellular data disabled (prevents iOS fallback to LTE)

## Emergency Recovery

If the tablet freezes or WebGL context is lost:

1. The `useContextRecovery` hook (Fix F14) will auto-reinit on return
2. If unresponsive: **Force restart** (Volume Up → Volume Down → Hold Side Button)
3. Re-enter Guided Access after restart
