import numpy as np


class RootCauseClassifier:

    def __init__(self,
                 jump_threshold_mm=30,
                 acc_threshold=300,
                 z_threshold_mm=20):

        self.jump_th = jump_threshold_mm
        self.acc_th = acc_threshold
        self.z_th = z_threshold_mm

    def classify(self, tcp_mm, acc_matrix, z_residual_mm):

        results = []

        # --- basic signal ---
        tcp_diff = np.linalg.norm(np.diff(tcp_mm, axis=0), axis=1)

        jump_flags = tcp_diff > self.jump_th
        acc_flags = np.max(np.abs(acc_matrix), axis=1) > self.acc_th
        z_flags = np.abs(z_residual_mm) > self.z_th

        total_frames = len(tcp_mm)

        # ============================================================
        # 1. Unit mismatch
        # ============================================================

        if np.mean(jump_flags) > 0.8:
            return [{
                "type": "Unit Mismatch",
                "confidence": "HIGH",
                "description": "Nearly all frames show abnormal TCP jumps. Likely DEG/RAD mismatch."
            }]

        # ============================================================
        # 2. Solver Instability
        # ============================================================

        for i in range(1, total_frames - 1):

            if jump_flags[i] and acc_flags[i] and z_flags[i]:
                results.append({
                    "frame": i,
                    "type": "Solver Instability",
                    "confidence": "HIGH",
                    "tcp_jump_mm": round(tcp_diff[i], 2),
                    "acc_peak": round(np.max(np.abs(acc_matrix[i])), 2),
                    "z_residual_mm": round(z_residual_mm[i], 2)
                })

        # ============================================================
        # 3. FK inconsistency
        # ============================================================

        for i in range(1, total_frames - 1):

            if jump_flags[i] and not acc_flags[i]:
                results.append({
                    "frame": i,
                    "type": "FK Inconsistency",
                    "confidence": "MEDIUM",
                    "description": "TCP jump without joint acceleration spike"
                })

        # ============================================================
        # 4. Coordinate inversion
        # ============================================================

        z = tcp_mm[:, 2]

        if np.mean(z) < 0 and np.max(z) < 0:
            results.append({
                "type": "Coordinate Inversion",
                "confidence": "MEDIUM",
                "description": "Z-axis appears inverted (all negative)"
            })

        # ============================================================
        # fallback
        # ============================================================

        if not results:
            results.append({
                "type": "No Critical Issue",
                "confidence": "LOW",
                "description": "No dominant instability pattern detected"
            })

        return results


# ============================================================
# pretty print
# ============================================================

def print_root_cause(results):

    print("\n" + "="*50)
    print("SIPA v2.3 Root Cause Classification")
    print("="*50)

    for r in results:

        if "frame" in r:
            print(f"\nFrame {r['frame']}")
        else:
            print("\nGlobal Diagnosis")

        print(f"Type: {r['type']}")
        print(f"Confidence: {r['confidence']}")

        if "description" in r:
            print(f"Info: {r['description']}")

        if "tcp_jump_mm" in r:
            print(f"TCP spike: {r['tcp_jump_mm']} mm")
            print(f"Joint acc: {r['acc_peak']} rad/s²")
            print(f"Z residual: {r['z_residual_mm']} mm")
