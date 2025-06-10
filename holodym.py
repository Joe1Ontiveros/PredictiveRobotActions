import argparse
import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dynamic_gestures.main_controller import MainController
from dynamic_gestures.utils import Drawer, Event, targets

# Global Variables
topic = '/hololens/camera/'
frame = None
bridge = CvBridge()
args = None
controller = None
drawer = None
debug_mode = False
image_received = False

def image_callback(msg):
    global frame, image_received
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_received = True
    except Exception as e:
        rospy.logwarn(f"Failed to convert image: {e}")
        frame = None

def process_frame(current_frame, controller, drawer, debug_mode):
    start_time = time.time()
    bboxes, ids, labels = controller(current_frame)
    if debug_mode:
        if bboxes is not None:
            bboxes = bboxes.astype(np.int32)
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
                gesture = targets[labels[i]] if labels[i] is not None else "None"
                cv2.rectangle(current_frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                cv2.putText(
                    current_frame,
                    f"ID {ids[i]} : {gesture}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(current_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if len(controller.tracks) > 0:
        count_of_zoom = 0
        thumb_boxes = []
        for trk in controller.tracks:
            if trk["tracker"].time_since_update < 1:
                if len(trk['hands']):
                    count_of_zoom += (trk['hands'][-1].gesture == 3)
                    thumb_boxes.append(trk['hands'][-1].bbox)
                    if len(trk['hands']) > 3 and [trk['hands'][-1].gesture, trk['hands'][-2].gesture, trk['hands'][-3].gesture] == [23, 23, 23]:
                        x, y, x2, y2 = map(int, trk['hands'][-1].bbox)
                        x, y, x2, y2 = max(x, 0), max(y, 0), max(x2, 0), max(y2, 0)
                        bbox_area = current_frame[y:y2, x:x2]
                        blurred_bbox = cv2.GaussianBlur(bbox_area, (51, 51), 10)
                        current_frame[y:y2, x:x2] = blurred_bbox

                if trk["hands"].action is not None:
                    drawer.set_action(trk["hands"].action)
                    trk["hands"].action = None
        if count_of_zoom == 2:
            drawer.draw_two_hands(current_frame, thumb_boxes)
    if debug_mode:
        current_frame = drawer.draw(current_frame)
    return current_frame

def run_inference():
    global controller, drawer, debug_mode
    use_ros = False
    try:
        rospy.init_node("gesture_recog_HL2", anonymous=True)
        rospy.Subscriber(topic, Image, image_callback)
        print("[INFO] ROS node initialized. Waiting for image topic...")
        # Wait for up to 4 seconds for the first image
        timeout = 4.0  # seconds
        start_wait = time.time()
        while not image_received and (time.time() - start_wait) < timeout:
            rospy.sleep(0.1)
        if image_received:
            print("[INFO] Received ROS image. Using ROS topic.")
            use_ros = True
        else:
            print("[ERROR] No image received from ROS topic. Please check if Hololens is publishing.")
            exit(1)
    except rospy.ROSException as e:
        print(f"[ERROR] ROS initialization failed: {e}")
        exit(1)
    # setup webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[WARN] Webcam not available. Continuing with ROS only...")
        cap = None

    while not rospy.is_shutdown():
        if frame is None:
            rospy.logwarn(f"No frame received from topic {topic} yet...")
            time.sleep(0.1)
            continue
        ros_frame = frame.copy()
        ros_frame = process_frame(ros_frame, controller, drawer, debug_mode)
        cv2.imshow("Hololens Feed", ros_frame)
        # webcam Frame (Optional)
        if cap is not None:
            ret, webcam_frame = cap.read()
            if ret:
                webcam_frame = cv2.flip(webcam_frame, 1)
                webcam_frame = process_frame(webcam_frame, controller, drawer, debug_mode)
                cv2.imshow("Webcam Feed", webcam_frame)
            else:
                print("[ERROR] Failed to grab frame from webcam")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def main():
    global args, controller, drawer, debug_mode

    parser = argparse.ArgumentParser(description="Run Gesture Recognition Demo")
    parser.add_argument(
        "--detector",
        default='dynamic_gestures/models/hand_detector.onnx',
        type=str,
        help="Path to detector onnx model"
    )
    parser.add_argument(
        "--classifier",
        default='dynamic_gestures/models/crops_classifier.onnx',
        type=str,
        help="Path to classifier onnx model",
    )
    parser.add_argument("--debug", required=False, action="store_true", help="Debug mode")
    args = parser.parse_args()

    controller = MainController(args.detector, args.classifier)
    drawer = Drawer()
    debug_mode = args.debug

    run_inference()

if __name__ == "__main__":
    main()

