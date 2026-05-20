## Installation guide
```bash
# aria glass environment
uv venv .venv -p 3.10
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install setuptools wheel
uv pip install -r requirements.txt --no-build-isolation
uv pip install projectaria_client_sdk==1.1.0 --no-cache-dir --prerelease=allow
uv pip install "ultralytics>=8.3"

# yolo+sam3 service
uv venv ~/venv/sam3_env -p 3.10
source ~/venv/sam3_env/bin/activate

uv pip install --upgrade pip
uv pip install setuptools wheel
uv pip install -r requirements_sam.txt
uv pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

When the glass is reset or paired to a new device, setup everything in the companion app first. Then follow the instructions below. 
## Initialization
```bash
aria-doctor

aria auth pair # check phone notification to confirm pairing
aria streaming install-certs
```

## Usage
Start streaming with aria glass sdk:
```bash
# subscribe first(needed in all scenarios)
python3 utils/streaming_start.py --interface wifi

# stop streaming
aria streaming stop --device-ip 192.168.8.117
```
In .venv:
```bash
# indicator key mapping, PAGEUP & B grabbed by evdev
python3 src/key_manager.py

# gaze detection, with mouse focus on gaze image output window, press top left button on the indicator to start collecting calibration data while focusing on center of marker, press once more to stop.
python3 -m projectaria_eyetracking.gaze_detect

# gaze projection on egocentric image with yolo detection and gaze score calculation
python3 -m src.gaze_rgb_visualizer --yolo --draw-gaze --participant AB12

# record audio, long press bottom right button
python3 -m src.audio_record --device-ip 192.168.8.117 --participant AB12
```
In ~/venv/sam3_env:
```bash
# YOLO+SAM3 service
python3 src/seg_service.py --visualize # argument for realsense segmentation image
```
Data collection and debug:
```bash
# capture data for training
python3 -m src.gaze_rgb_visualizer --capture

# ros bag
python3 src/bag_record.py --participant AB12

# generate short video for aria rgb camera based on specific timestamp for arbitrary participant and trial
python3 src/clip_from_frames.py --participant AB12 --trial 04 --timestamp 1779296845648620199 --window 10
```

## Misc
```bash
# streaming demo with glfw window visualization, in .venv
python3 utils/device_stream.py --interface wifi --device-ip 192.168.8.117 --update_iptables

# visualize point cloud to be sent, in sam3_env
python3 utils/pc_viz.py 

# test command
ros2 param set /seg_service target_label 'tomato'
ros2 service call /seg/infer std_srvs/srv/Trigger '{}'
```
## Multi-device ROS config on local network
Install dependency:
```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-rmw-fastrtps-cpp
```

Verify `ip a` to check network interface name accordingly. Set up for both devices. Add cyclonedds config to `~/.ros/cyclonedds.xml`.
For both workstation and laptop:
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

#### Setup DDS config on workstation and laptop
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=10
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
```