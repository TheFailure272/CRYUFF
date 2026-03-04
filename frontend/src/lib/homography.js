/**
 * C.R.U.Y.F.F. — PTZ Homography (Fix F13 + Fix F17)
 *
 * Fix F17: You cannot linearly interpolate a raw 4×4 matrix — it shears
 * the affine basis during simultaneous pan+zoom. Instead, decompose into:
 *   - Translation (Vector3) → FRI damp
 *   - Rotation (Quaternion) → SLERP
 *   - Scale (Vector3) → FRI damp
 * Then recompose on every frame.
 */

import * as THREE from 'three';
import { ANIM } from './constants';

// Reusable temporaries (zero allocation in hot path)
const _targetPos = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();
const _targetScale = new THREE.Vector3(1, 1, 1);

export class PTZHomography {
    constructor() {
        // Current interpolated TRS
        this._pos = new THREE.Vector3();
        this._quat = new THREE.Quaternion();
        this._scale = new THREE.Vector3(1, 1, 1);

        // Target TRS (from latest telemetry)
        this._tPos = new THREE.Vector3();
        this._tQuat = new THREE.Quaternion();
        this._tScale = new THREE.Vector3(1, 1, 1);

        // Composed matrix
        this._matrix = new THREE.Matrix4();
        this._active = false;
    }

    /**
     * Update from incoming telemetry payload.
     * Expected: data.camera_matrix = [16 floats] (column-major 4×4)
     *
     * @param {Object} data
     */
    update(data) {
        const cm = data?.camera_matrix;
        if (!cm || !Array.isArray(cm) || cm.length !== 16) return;

        // Decompose the raw 4×4 into Translation, Rotation, Scale
        // This is the Fix F17 correction — never LERP the raw matrix.
        const m = new THREE.Matrix4().fromArray(cm);
        m.decompose(_targetPos, _targetQuat, _targetScale);

        this._tPos.copy(_targetPos);
        this._tQuat.copy(_targetQuat);
        this._tScale.copy(_targetScale);
        this._active = true;
    }

    /**
     * Step one frame: FRI-damp position & scale, SLERP rotation.
     *
     * @param {number} delta - Frame delta (seconds)
     * @returns {THREE.Matrix4}
     */
    step(delta) {
        if (!this._active) return this._matrix;

        const alpha = 1 - Math.exp(-ANIM.DAMP_LAMBDA * delta);

        // Translation: FRI damp
        this._pos.lerp(this._tPos, alpha);

        // Rotation: SLERP (Fix F17 — prevents affine shearing)
        this._quat.slerp(this._tQuat, alpha);

        // Scale: FRI damp
        this._scale.lerp(this._tScale, alpha);

        // Recompose
        this._matrix.compose(this._pos, this._quat, this._scale);
        return this._matrix;
    }

    /**
     * Apply to a Three.js camera.
     *
     * @param {THREE.Camera} camera
     * @param {number} delta
     */
    applyToCamera(camera, delta) {
        if (!this._active) return;
        const m = this.step(delta);
        camera.projectionMatrix.copy(m);
        camera.projectionMatrixInverse.copy(m).invert();
    }

    /**
     * Project a normalised pitch coordinate through the homography.
     *
     * @param {number} x - Pitch x [0,1]
     * @param {number} y - Pitch y [0,1]
     * @returns {{sx: number, sy: number}}
     */
    project(x, y) {
        const v = new THREE.Vector4(x, y, 0, 1);
        v.applyMatrix4(this._matrix);
        if (Math.abs(v.w) < 1e-6) return { sx: 0, sy: 0 };
        return { sx: v.x / v.w, sy: v.y / v.w };
    }

    get isActive() {
        return this._active;
    }
}
