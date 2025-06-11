# How to run base model and action-gesture guessing on the HOLOLENS 
The model is already trained and usable by running dynamic_hololens.py if you are using a hololens that is TCP connected to your PC and is outputting an image topic of the camera sensor (/hololens/ab_image/). You should see the outgoing feed under (/hololens/outgoing_imge)

# How to run on your own PC
You can still use the model and LTSM on your own pc by running run_pc_demo.py and having a bluetooth camera attached (accessible by opencv at index 0) .

You will see the gesture read by dynamic_gestures on the left (sequence of 3) and the predicted outcome on the right of the square sourrounding your hand. 
It is also printed to terminal


# Re-training:
If you'd like to retrain, feed a .csv/edit the path in LTSM_BASE.PY to the desired training set, ensure you match columns names/purposes of the existing datasets
(3 gestures | desired prediction/output ) 



# Unity Setup
Ensure there exists some method of getting images from the 
HoloLens camera & depth sensor to this script. 
We will use this to classify user gestures that are fed to our 
model for responses in the headset as well
See @UnityRoboticsHub on the HL2 & Unity ROS TCP Connector on the backend

# Software Hirachy 

The Listener/main class (dynamic_hololens.py)
 Takes the images from the HoloLens (or any image source)
 and feeds it into dynamic_gesture model to return a gesture recognition at that timestamp (We iterate/predict every 5 seconds )

Ingest.py takes in additional data like the robots position
 the users position & gaze, and has room for other metrics
The desired tasks for robots/gesture to task are the following: 


"STOP/PALM: DELIMITER/STOPPER" , 
"1,2,3,4 MEANS NUMBER OF TIMES TO REPEAT MOVE/ROTATE COMMAND",
"POINTLEFT:ROTATELEFT","POINTRIGHT:ROTATERIGHT",  
"THUMBSUP:MOVEFORWARD", "THUMBSDOWN:MOVEBACKWARD" 


# SETUP : 
in root: 
'''bash
# Create virtual env by conda or venv
conda create -n dynamic_gestures_HL python=3.9 -y
conda activate dynamic_gestures_HL
# Install requirements
pip install -r dynamic_gestures_HL/requirements.txt
pip install -r requirements.txt
'''
