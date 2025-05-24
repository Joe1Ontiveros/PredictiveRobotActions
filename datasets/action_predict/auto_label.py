import os
import json
from collections import defaultdict, Counter

from dynamic_gestures.maincontroller import gesture  # Assuming gesture() yields gesture names

# Define gesture-to-action mapping and action chains
GESTURE_ACTIONS = {
    "POINTLEFT": "rotate_left",
    "POINTRIGHT": "rotate_right",
    "THUMBS_UP":"forward",
    "THUMBS_DOWN":"down",
    "FIST": "DELIM",
    "PALM": "STOP",
    "PEACE": "2",
    "littlefinger":"1",
    # Add more mappings as needed
}

# Define possible action chains (example)
ACTION_CHAINS = [
    ["rotate_left", "rotate_left"],
    ["rotate_right", "rotate_right"],
    ["move_forward", "move_forward"],
    ["move_backward", "move_backward"],
    ["rotate_left", "move_forward"],
    ["rotate_right", "move_forward"],
    # Add more chains as needed
]

DELIMITER = "<DELIM>"

def clean_gesture_sequence(gesture_seq):
    """Remove noise and invalid gestures."""
    return [g for g in gesture_seq if g in GESTURE_ACTIONS]

def gestures_to_actions(gesture_seq):
    """Map gestures to actions."""
    return [GESTURE_ACTIONS[g] for g in gesture_seq]

def markov_chain_labeling(action_seq):
    """Label each action with its next likely action (Markov model)."""
    transitions = defaultdict(Counter)
    for i in range(len(action_seq) - 1):
        transitions[action_seq[i]][action_seq[i+1]] += 1
    # For each action, predict the most likely next action
    predictions = {}
    for action, nexts in transitions.items():
        predictions[action] = nexts.most_common(1)[0][0]
    return predictions

def label_images(image_dir, output_json):
    """
    For each image, label with:
      - gesture
      - action
      - action_chain (if part of a chain)
      - next_predicted_gesture/action
    """
    # Simulate gesture output for each image (replace with actual gesture() call)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    gesture_seq = [gesture(os.path.join(image_dir, f)) for f in image_files]  # gesture() should return gesture name

    # Clean and map to actions
    gesture_seq = clean_gesture_sequence(gesture_seq)
    action_seq = gestures_to_actions(gesture_seq)

    # Markov model for next action prediction
    next_action_pred = markov_chain_labeling(action_seq)

    # Build action chains
    chains = []
    for chain in ACTION_CHAINS:
        for i in range(len(action_seq) - len(chain) + 1):
            if action_seq[i:i+len(chain)] == chain:
                chains.append((i, i+len(chain), chain))

    # Label images
    labels = []
    for idx, (img, gesture_name, action) in enumerate(zip(image_files, gesture_seq, action_seq)):
        # Find if part of a chain
        chain_label = None
        for start, end, chain in chains:
            if start <= idx < end:
                chain_label = DELIMITER.join(chain)
                break
        # Predict next gesture/action
        next_action = next_action_pred.get(action, None)
        next_gesture = None
        if next_action:
            for g, a in GESTURE_ACTIONS.items():
                if a == next_action:
                    next_gesture = g
                    break
        labels.append({
            "image": img,
            "gesture": gesture_name,
            "action": action,
            "action_chain": chain_label,
            "next_predicted_action": next_action,
            "next_predicted_gesture": next_gesture
        })

    # Save labels for multi-model training
    with open(output_json, "w") as f:
        json.dump(labels, f, indent=2)

if __name__ == "__main__":
    image_dir = "images"  # Directory containing images
    output_json = "labels.json"
    label_images(image_dir, output_json)