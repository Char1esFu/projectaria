## Installation guide
```bash
uv venv .venv -p 3.10
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install projectaria_client_sdk==1.1.0 --no-cache-dir --prerelease=allow
```

## Initialization
```bash
aria-doctor

aria auth pair # check phone notification to confirm pairing
aria streaming install-certs
```

## Usage
```bash
# streaming demo with glfw window. Run this first to give sudo access, otherwise streaming will fail later.
python3 utils/device_stream.py --interface usb --update_iptables

# subscribe first(needed in all scenarios)
aria streaming start --interface usb # optional: --use-ephemeral-certs

# glass localization, specify arbitrary aruco marker id(s) in the setup
source /opt/ros/humble/setup.bash
python3 -m src.aruco_localization --marker-ids 1 2 # list all used marker ids in argument

# gaze detection
python3 -m projectaria_eyetracking.gaze_detect --pitch-bias 0.1 # calibrate this value by looking straight ahead and check pitch output. Compensate to 0 for each user

# publish gaze table intersection
python3 src/gaze_intersection_node.py --visualize

```

## Utils
```bash
# estimate pose with IMU data and Madgwick filter as failsafe
python3 -m src.imu_pose --imu-index 1 # or 0

# streaming demo with individual message
python3 -m utils.streaming_subscribe
aria streaming stop
```