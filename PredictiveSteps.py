from collections import defaultdict, Counter, deque
import json
from typing import Optional

# get inputs and call our trained model for the next predicted 
# gestures 
class PredictGestures:
    def __init__(self, model: Optional[str] = None):
        self.gesture_history = deque(maxlen=10)
        self.gaze_history = deque(maxlen=10)
        self.imu_history = deque(maxlen=10)
        self.robot_pose_history = deque(maxlen=10)

        self.transition_counts = defaultdict(Counter)
        self.transition_probs = {}
        self.model_path = model

        print(f"[Init] PredictGestures initialized. Model: {model}")

    def store_recog(self, incoming: str):
        """Store the incoming recognized gesture"""
        print(f"[Store] Gesture: {incoming}")
        if self.gesture_history:
            prev = self.gesture_history[-1]
            self.transition_counts[prev][incoming] += 1
        self.gesture_history.append(incoming)

    def store_gaze(self, incoming):
        """Store gaze vector or target"""
        print(f"[Store] Gaze: {incoming}")
        self.gaze_history.append(incoming)

    def store_IMU(self, incoming):
        """Store IMU or positional movement"""
        print(f"[Store] IMU: {incoming}")
        self.imu_history.append(incoming)

    def store_robot(self, incoming_PoseStamp, incoming_TF):
        """Store robot pose information relative to user"""
        self.robot_pose_history.append({
            'pose': incoming_PoseStamp,
            'tf': incoming_TF
        })
        print(f"[Store] Robot pose stored.")

    def _build_transition_model(self):
        """Build or update the gesture transition probabilities"""
        self.transition_probs = {
            g1: {g2: c / sum(counter.values()) for g2, c in counter.items()}
            for g1, counter in self.transition_counts.items()
        }
        print(f"[Model] Transition probabilities updated.")

    def _predict_next(self, current: str) -> str:
        """Predict next gesture based on transition probabilities"""
        if current not in self.transition_probs:
            return "UNKNOWN"
        next_gestures = self.transition_probs[current]
        return max(next_gestures, key=next_gestures.get)

    def next_n_steps(self, n: int, op: Optional[str] = None) -> dict:
        """Predict next `n` gestures"""
        if op:
            print(f"[Predict] Operant mode: {op}")

        if not self.gesture_history:
            return {"current": None, "predicted": []}

        self._build_transition_model()

        current = self.gesture_history[-1]
        predicted = []

        for _ in range(n):
            next_step = self._predict_next(current)
            predicted.append(next_step)
            current = next_step  # feed it back for next prediction

        output = {
            "current": self.gesture_history[-1],
            "predicted": predicted
        }

        print(f"[Output] {json.dumps(output, indent=2)}")
        return output
