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
# streaming demo with glfw window
python3 utils/device_stream.py --interface usb --update_iptables

# streaming demo with individual message
aria streaming start --interface usb # optional: --use-ephemeral-certs
python3 utils/streaming_subscribe.py
aria streaming stop

# streaming with eye tracking inference
python -m projectaria_eyetracking.streaming_eye_inference_demo

# glass localization
python3 utils/aruco_localization.py --ros2-publish
```