
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

These metrics are utilized for determining the end task goal

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