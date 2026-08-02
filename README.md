# Project Aria Eye Tracking

## Install

Aria environment:

```bash
uv venv .venv -p 3.10
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements.txt --no-build-isolation
uv pip install projectaria_client_sdk==1.1.0 --no-cache-dir --prerelease=allow
uv pip install "ultralytics>=8.3"
```

YOLO + SAM3 environment:

```bash
uv venv ~/venv/sam3_env -p 3.10
source ~/venv/sam3_env/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements_sam.txt
uv pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Initialize Aria

Complete device setup in the companion app, then run:

```bash
aria-doctor
aria auth pair
aria streaming install-certs
```

## Stream

```bash
aria streaming start --interface wifi --device-ip 192.168.8.117 --use-ephemeral-certs --profile profile18
aria streaming start --interface usb --use-ephemeral-certs --profile profile5
aria streaming stop --device-ip 192.168.8.117
```

## Run

Activate `.venv`, then start the key manager and gaze detector:

```bash
python3 src/key_manager.py
python3 -m projectaria_eyetracking.gaze_detect
```

Run online gaze-label recording and variance selection:

```bash
python3 -m src.gaze_rgb_visualizer \
  --participant test02 \
  --rgb-buffer-delay-frames 5 \
  --rgb-timestamp-source hardware \
  --boundary-radius 15 \
  --gaze-var-window 3 \
  --gaze-var-threshold 1.5e-3 \
  --gaze-var-top 1 \
  --gaze-var-force-endpoint-points 1
```

Add `--hide-excluded` to omit endpoint windows from the variance timeline. It does not change selection or the published result.

Each recording stores raw/in-filled detections in `gaze_labels.json`, selected timestamps in `variance_selected/scores.json`, and the normalized result in `published.json`.

Re-run the variance-to-published pipeline offline without modifying `gaze_labels.json`:

```bash
python3 -m src.offline_gaze_label \
  recordings/test02/31 \
  --window 3 \
  --threshold 1.5e-3 \
  --top 1 \
  --boundary-radius 15 \
  --force-endpoint-points 1 \
  --hide-excluded \
  --pretty \
  --details
```

Refresh detection in-fill for an older recording:

```bash
python3 -m src.detection_infill recordings/test02/33 --min-label-observations 2 --write
```

Audio input

```bash
python3 -m src.audio_record --participant AB12 --source local
```

Run the YOLO + SAM3 service from `~/venv/sam3_env`:

```bash
python3 src/seg_service.py --visualize
```

## Utilities

```bash
python3 -m src.gaze_rgb_visualizer --capture
python3 src/bag_record.py --participant AB12
python3 utils/device_stream.py --interface wifi --device-ip 192.168.8.117 --update_iptables
python3 utils/pcd_viz.py
```

Segmentation test:

```bash
ros2 param set /seg_service target_label tomato
ros2 service call /seg/infer std_srvs/srv/Trigger '{}'
```

## ROS network

Install Fast DDS on both computers:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-rmw-fastrtps-cpp
```

Create `~/.ros/cyclonedds.xml` on both computers and set the network interface:

```xml
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="wlp5s0"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
```

Environment:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=10
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
```
