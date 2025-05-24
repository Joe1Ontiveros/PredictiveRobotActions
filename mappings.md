| Signal       | Format Example                       | Purpose                 |
| ------------ | ------------------------------------ | ----------------------- |
| Gesture      | `"SWIPE_LEFT"`                       | Input sequence          |
| Gaze Target  | `"button_A"` or vector `[x, y, z]`   | Spatial intent/context  |
| IMU          | `{"accel": [...], "gyro": [...]}`    | Motion cues             |
| Robot Pose   | `PoseStamped + TF` (relative coords) | Spatial relationship    |
| Output Label | `"TAP"` (next step or gesture label) | Ground truth to predict |
