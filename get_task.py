GESTURE_ACTIONS = {
    "POINTLEFT": "rotate_left",
    "POINTRIGHT": "rotate_right",
    "THUMBS_UP": "forward",
    "THUMBS_DOWN": "down",
    "FIST": "DELIM",  # delimiter for multi-step command
    "PALM": "STOP",   # command separator
    "PEACE": "2",
    "littlefinger": "1"
}

DELIMITER = "<DELIM>"
STOP_TOKEN = "<STOP>"
# with the incoming feed we get/determine the path to follow algortihmically 
 
def gesture_sequence_to_action_plan(gestures):
    """
    Interprets gesture sequence into a structured robot action plan.
    Handles directional commands, intensity, delimiters, and stops.
    """
    actions = []
    i = 0
    while i < len(gestures):
        g = gestures[i]

        # Handle STOP
        if g == "PALM":
            actions.append(STOP_TOKEN)
            i += 1
            continue

        # Handle Directional Rotations
        if g in ("POINTLEFT", "POINTRIGHT") and i + 2 < len(gestures):
            if gestures[i+1] == "FIST":  # delimiter pattern
                magnitude = GESTURE_ACTIONS.get(gestures[i+2], "1")
                dir_action = GESTURE_ACTIONS[g]
                actions.append(f"{dir_action}_{magnitude}")
                i += 3
                continue

        # Handle Forward/Down movement
        if g in ("THUMBS_UP", "THUMBS_DOWN") and i + 2 < len(gestures):
            if gestures[i+1] == "FIST":
                magnitude = GESTURE_ACTIONS.get(gestures[i+2], "1")
                base_action = GESTURE_ACTIONS[g]
                actions.append(f"move_{base_action}_{magnitude}")
                i += 3
                continue

        # Fallback: just skip unknown patterns
        i += 1

    return actions
