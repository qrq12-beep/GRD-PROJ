import numpy as np

class PoseFightDetector:
    def __init__(self):
        self.prev_keypoints = None
        self.confirm_frames = 0

    def detect(self, results):
        if results[0].keypoints is None:
            return False

        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) < 2:
            self.prev_keypoints = keypoints
            return False

        fight_score = 0

        # ---- 1. Distance check ----
        center1 = np.mean(keypoints[0], axis=0)
        center2 = np.mean(keypoints[1], axis=0)

        distance = np.linalg.norm(center1 - center2)

        if distance < 200:
            fight_score += 1

        # ---- 2. Arm speed check ----
        if self.prev_keypoints is not None:
            for i in range(2):
                wrist_now = keypoints[i][9:11]  # wrists
                wrist_prev = self.prev_keypoints[i][9:11]

                movement = np.linalg.norm(wrist_now - wrist_prev)

                if movement > 40:
                    fight_score += 1

        self.prev_keypoints = keypoints

        # ---- 3. Multi-frame confirmation ----
        if fight_score >= 2:
            self.confirm_frames += 1
        else:
            self.confirm_frames = 0

        return self.confirm_frames >= 3
