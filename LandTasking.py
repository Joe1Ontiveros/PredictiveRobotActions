import ingest
import numpy as np
import rospy
import openai
import json

GESTURE_TASK_MAP = {
    "wave": "move_forward",
    "fist": "stop",
    "point_left": "turn_left",
    "point_right": "turn_right",
    "thumbs_up": "confirm",
    "thumbs_down": "cancel"
}

class AutonomousAgent:
    def __init__(self):
        self.llm_attempts = 0
        self.max_attempts = 3

    def parse_LLM_designation(self, gesture):
        # Try direct mapping first
        if gesture in GESTURE_TASK_MAP:
            return {"function": GESTURE_TASK_MAP[gesture], "args": {}}

        # Otherwise, ask LLM for a function call (simulate function-calling)
        prompt = (
            f"Given the gesture '{gesture}', output a JSON object with the function to call and any arguments. "
            "Example: {\"function\": \"move_forward\", \"args\": {}}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a robot task planner."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response['choices'][0]['message']['content'].strip()
            # Try to parse as JSON
            try:
                call = json.loads(content)
                return call
            except Exception:
                # Fallback: treat as function name string
                return {"function": content, "args": {}}
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def request_user(self, TextDesc: str) -> bool:
        print(f"Requesting user approval for: {TextDesc}")
        return True

    def prune_prompt(self, llm_call):
        # Ensure function name is safe and lowercased
        func = llm_call.get("function", "").strip().lower()
        args = llm_call.get("args", {})
        return {"function": func, "args": args}

    def check_prompt(self, pruned_call):
        allowed = {"move_forward", "stop", "turn_left", "turn_right", "confirm", "cancel"}
        return pruned_call["function"] in allowed

    def format_prompt(self, pruned_call):
        func = getattr(self, pruned_call["function"], None)
        return func, pruned_call["args"]

    def abstract_detail(self, task):
        descriptions = {
            "move_forward": "Move the robot forward.",
            "stop": "Stop the robot.",
            "turn_left": "Turn the robot left.",
            "turn_right": "Turn the robot right.",
            "confirm": "Confirm the current action.",
            "cancel": "Cancel the current action."
        }
        return descriptions.get(task, f"Perform task: {task}")

    # Example robot task functions
    def move_forward(self, **kwargs):
        print("Robot moving forward (ROS call here)")

    def stop(self, **kwargs):
        print("Robot stopping (ROS call here)")

    def turn_left(self, **kwargs):
        print("Robot turning left (ROS call here)")

    def turn_right(self, **kwargs):
        print("Robot turning right (ROS call here)")

    def confirm(self, **kwargs):
        print("Action confirmed (ROS call here)")

    def cancel(self, **kwargs):
        print("Action cancelled (ROS call here)")

    def handle_gesture(self, gesture):
        for attempt in range(self.max_attempts):
            llm_call = self.parse_LLM_designation(gesture)
            if not llm_call:
                continue
            pruned = self.prune_prompt(llm_call)
            if not self.check_prompt(pruned):
                print(f"Rejected unsafe or unknown task: {pruned['function']}")
                continue
            desc = self.abstract_detail(pruned["function"])
            if self.request_user(desc):
                func, args = self.format_prompt(pruned)
                if func:
                    func(**args)  # <-- LLM can now call with arguments!
                    print(f"Executed: {pruned['function']} with args {args}")
                    return True
                else:
                    print(f"No function found for: {pruned['function']}")
            else:
                print("User did not approve.")
                return False
        print("Failed to infer/approve a valid task after 3 attempts.")
        return False

# Example usage:
if __name__ == "__main__":
    agent = AutonomousAgent()
    # Simulate incoming gesture
    gesture = "wave"
    agent.handle_gesture(gesture)