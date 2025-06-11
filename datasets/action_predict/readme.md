# 300-Image-label setup
 # sketch for IMAGE-BASED model, settled for predictive LTSM due to time constraints

*Test the following*
We will take a basic set of 100 manually labeled instances of gestures related to an action
# hand out is stop
 
# fist is define distance
20 images pre-labelled, 
1. Predictive images will be up,down,left,right 
2. we want the model to be trained to recognize that a NUMBER comes after these gestures, so more training done here on the 1,2,3 finger count  

# thumbs up is move forward
we want it to predict that fist follows this, and that prior it, we can assume rotate is called (user will want to rotate before moving )
1. so train it on recognizing point left or point right is next based on the thumbs



# thumbs down is move backward 
# point left is rotate left 
# point right is rotate right 


# Sequence chaining and adaptive learning
1. Our model should be able to learn from the user's past actions and potentially suggest a course/plan of actions based on whats happened so far
Say I have told the robot to move forward rotate 35* 2 times already, our model should be able to reasonably predict one of the actions to be that we want to drive in a circle, so it can suggest another 35* forward command or 35* forward x2 to make a full rotation


| Image/Sequence                | Label                      | Note                |
| ----------------------------- | -------------------------- | ------------------- |
| `hand_out.jpg`                | `STOP`                     | Stop Execution      |
| `fist.jpg`                    | `DEFINE_DISTANCE`          | Followed by number  |
| `thumbs_up.jpg`               | `MOVE_FORWARD`             | Expect "fist" after |
| `thumbs_down.jpg`             | `MOVE_BACKWARD`            | Expect "fist" after |
| `point_left.jpg`              | `ROTATE_LEFT`              | Used before moving  |
| `point_right.jpg`             | `ROTATE_RIGHT`             | Used before moving  |
| `1.jpg`–`3.jpg`               | `DISTANCE_1`–`DISTANCE_3`  | Numeric context     |


100 actual manually made and referenced images 
100-200 inverted/cropped/edited instances of the base image