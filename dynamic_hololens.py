import argparse
import time
import os
import torch
import pandas as pd
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from m2 import MainController
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
global topic
topic = '/hololens/camera/'

output_pub = None
bridge = None

# Load LSTM model and vocabularies ONCE
lstm_model_path = "trained_lstm_model.pth"
csv_path = "datasets/action_predict/Balanced_Gesture_Command_Dataset.csv"
input_len = 3
hidden_size = 128
num_layers = 2
num_classes = 10

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
    global frame, output_pub, bridge, gesture_history
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logwarn(f"CV Bridge error: {e}")
        return

    controller = MainController(args.detector, args.classifier)
    drawer = Drawer()
    debug_mode = args.debug

    bboxes, ids, labels = controller(frame)
    gesture = None
    if bboxes is not None:
        bboxes = bboxes.astype(np.int32)
        for i in range(bboxes.shape[0]):
            box = bboxes[i, :]
            gesture = targets[labels[i]] if labels[i] is not None else "None"
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

    # Predict action every time we have 3 gestures
    predicted_action = "None"
    if len(gesture_history) == 3 and lstm_model is not None:
        predicted_action = predict_action(lstm_model, gesture_vocab, action_vocab, gesture_history)
        cv2.putText(
            frame,
            f"Predicted Action: {predicted_action}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        print(f"Predicted action for {gesture_history}: {predicted_action}")

    # Draw overlays and publish
    frame = drawer.draw(frame)
    try:
        output_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output_pub.publish(output_msg)
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")

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

    rospy.Subscriber("/hololens/camera/", Image, image_callback)
    rospy.spin()

if __name__ == "__main__":
    ros_main()