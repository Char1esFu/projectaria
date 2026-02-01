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
# streaming demo with glfw window(run this first to give sudo access, otherwise streaming will fail later)
python3 utils/device_stream.py --interface usb --update_iptables

# estimate pose with IMU data and Madgwick filter as failsafe
python3 -m src.imu_pose --imu-index 1 # or 0

# subscribe first(needed in all scenarios)
aria streaming start --interface usb # optional: --use-ephemeral-certs

# streaming demo with individual message
python3 utils/streaming_subscribe.py
aria streaming stop

# streaming with eye tracking inference
python3 -m projectaria_eyetracking.streaming_eye_tracking

# glass localization, specify arbitrary aruco marker id(s) in the setup
source /opt/ros/humble/setup.bash
python3 -m src.aruco_localization --marker-ids 5
```