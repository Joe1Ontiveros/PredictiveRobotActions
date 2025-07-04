import argparse
import os
import torch
import pandas as pd
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from m2 import MainController  # <-- Use m2.MainController for parity with your working demo
from utils import Drawer, Event, targets

# --- LSTM Model and Utilities ---

class LSTM(torch.nn.Module):
    def __init__(self, input_len, hidden_size, num_class, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = torch.nn.Linear(hidden_size, num_class)

    def forward(self, X):
        device = X.device
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=device)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out

def build_vocab_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    gesture_cols = ['gesture_1', 'gesture_2', 'gesture_3']
    gesture_set = set()
    for col in gesture_cols:
        gesture_set.update(df[col].unique())
    gesture_vocab = {g: i for i, g in enumerate(sorted(gesture_set))}
    action_vocab = list(pd.factorize(df['next_action'])[1])
    return gesture_vocab, action_vocab

def predict_action(model, gesture_vocab, action_vocab, gestures):
    encoded = [gesture_vocab.get(g, -1) for g in gestures]
    if -1 in encoded:
        print("Unknown gesture in input:", gestures)
        return "unknown"
    X = torch.tensor(encoded, dtype=torch.float32).reshape(1, 1, 3)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            X = X.to("cuda")
            model = model.to("cuda")
        output = model(X)
        pred_idx = output.argmax(dim=1).item()
        pred_action = action_vocab[pred_idx]
        return pred_action

# ROS/HL2 setup
topic = '/hololens/ab_image/'

output_pub = None
bridge = None
args = None

# Load LSTM model and vocabularies ONCE
lstm_model_path = "trained_lstm_model_ALL.pth"
csv_path = "datasets/action_predict/DDR2.csv"

# Dynamically determine num_classes from the CSV to match training
df = pd.read_csv(csv_path)
if df.shape[1] == 4:
    df.columns = ['gesture_1', 'gesture_2', 'gesture_3', 'next_action']
    num_classes = df['next_action'].nunique()
else:
    raise ValueError("CSV must have 4 columns (3 gestures + 1 action/command)")

input_len = 3
hidden_size = 256  # <-- match your training parameter!
num_layers = 2

gesture_vocab, action_vocab = build_vocab_from_csv(csv_path)
lstm_model = LSTM(input_len, hidden_size, num_classes, num_layers)
if os.path.exists(lstm_model_path):
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location="cpu"))
    lstm_model.eval()
    print("Loaded trained LSTM model for action prediction.")
else:
    print("Warning: Trained LSTM model not found. Action prediction will not work.")
    lstm_model = None

gesture_history = []
def image_callback(msg):
    global output_pub, bridge, gesture_history, args
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logwarn(f"CV Bridge error: {e}")
        return

    controller = MainController(args.detector, args.classifier)
    drawer = Drawer()
    debug_mode = args.debug

    # --- FIX: Use controller as in your working OpenCV demo ---
    bboxes, ids, labels = controller(frame)
    detected_gestures = []
    if bboxes is not None and labels is not None:
        bboxes = bboxes.astype(np.int32)
        for i in range(bboxes.shape[0]):
            box = bboxes[i, :]
            gesture = targets[labels[i]] if labels[i] is not None else "None"
            detected_gestures.append(gesture)
            print(f"[DEBUG] Recognized gesture: {gesture}")
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(
                frame,
                f"ID {ids[i]} : {gesture}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            if gesture != "None":
                gesture_history.append(gesture)
                if len(gesture_history) > 3:
                    gesture_history = gesture_history[-3:]
    else:
        print("[DEBUG] No gestures detected in this frame.")

    # Predict action every time we have 3 gestures
    predicted_action = "None"
    if len(gesture_history) == 3 and lstm_model is not None:
        predicted_action = predict_action(lstm_model, gesture_vocab, action_vocab, gesture_history)
        print(f"(CURRENT GESTURE SET {gesture_history}, PREDICTED_ACTION: {predicted_action})")
    else:
        print(f"(CURRENT GESTURE SET {gesture_history}, PREDICTED_ACTION: None)")

    # Show both detected gestures and predicted action in the output visual
    display_gestures = ', '.join(detected_gestures) if detected_gestures else "None"
    display_text = f"Gestures: [{display_gestures}] | Predicted: {predicted_action}"
    cv2.putText(
        frame,
        display_text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
    )

    # Draw overlays and publish
    frame = drawer.draw(frame)
    try:
        output_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output_pub.publish(output_msg)
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")

    # Show the outgoing feed in a window
    cv2.imshow("Outgoing Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[DEBUG] Exiting live feed window.")
        rospy.signal_shutdown("User requested shutdown.")

def ros_main():
    global output_pub, bridge, args
    rospy.init_node("gesture_recog_HL2")
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument(
        "--detector",
        default='models/hand_detector.onnx',
        type=str,
        help="Path to detector onnx model"
    )
    parser.add_argument(
        "--classifier",
        default='models/crops_classifier.onnx',
        type=str,
        help="Path to classifier onnx model",
    )
    parser.add_argument("--debug", required=False, action="store_true", help="Debug mode")
    args = parser.parse_args()
    output_pub = rospy.Publisher('/hololens/processed_image', Image, queue_size=1)
    global bridge
    bridge = CvBridge()
    rospy.Subscriber(topic, Image, image_callback)
    print("[DEBUG] ROS node started, waiting for images...")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ros_main()
    
    
    # current issues, cannot read gestures thus cannot feed to model, but we are succesfully capturing hololens and overlaying the feed 
    # fix that and then we are good to submit 