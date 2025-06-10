import cv2 as opencv
import numpy as np
import PIL as p
import openai
import rospy
from std_msgs.msg import String
import LandTasking
from main_controller import PREDICT_GESTURE_200

# actual model call here:
def next_(cont: list) -> str:
    """
    Predict the next gesture using the PREDICT_GESTURE_200 model.
    If the model fails, fallback to the Markov model.
    """
    try:
        # Assume PREDICT_GESTURE_200 is a function or model you have imported
        # and it takes the context list as input and returns a prediction.
        prediction = PREDICT_GESTURE_200(cont)
        if prediction is not None:
            return prediction
    except Exception as e:
        print(f"Error in PREDICT_GESTURE_200: {e}")
    # fallback to Markov model if prediction fails
    return fallback_Markov(len(cont), cont)


global context 
context = []
def callback_ModelOut(gesture:str):
    # get the dynamic gestures model output for the last n-calls/steps
    if gesture != "RESET":
        context.append(gesture) # setup context accordingly
    else:
        context = []
    if gesture is None:
        print("No gesture from MODEL ONN passed in, skipping/exiting...")
        return fallback_Markov(context)
    
    context = handle_errors(context)
    if len(context) > 2: 
        # apply the predicitve model and serve
        res = next_(context)
        callback_hololensOut("AI:"+res)
        # pass to LLM to call a robot action
        LandTasking.AutonomousAgent.format_prompt(res,context)
    else:
        return None # model needs at least 2 to predict 
def handle_errors(gestures):
    
    return gestures


def callback_hololensOut(gesture:str):
    pub = rospy.Publisher('hololens/outgoing/gesturerecog', String, queue_size=10)
    rospy.init_node('gesture_publisher', anonymous=True)
    msg = String()
    msg.data = gesture
    pub.publish(msg)

def fallback_Markov(n: int, gestures):
    # Given the last n gestures, predict the next gesture using a simple Markov model.
    if not gestures or len(gestures) < 2:
        return None
    # Build transition counts
    transitions = {}
    for i in range(len(gestures) - 1):
        curr = gestures[i]
        nxt = gestures[i + 1]
        if curr not in transitions:
            transitions[curr] = {}
        transitions[curr][nxt] = transitions[curr].get(nxt, 0) + 1
    # Use the last gesture to predict the next
    last_gesture = gestures[-1]
    if last_gesture not in transitions:
        return None
    # Pick the most probable next gesture
    next_gestures = transitions[last_gesture]
    predicted = max(next_gestures, key=next_gestures.get)
    return predicted
