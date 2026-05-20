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
In .venv with aria glass sdk:
```bash
# streaming demo with glfw window. Run this first to give sudo access, otherwise streaming will fail later.
python3 utils/device_stream.py --interface wifi --device-ip 192.168.8.117 --update_iptables

# subscribe first(needed in all scenarios)
aria streaming start --interface wifi --device-ip 192.168.8.117 --use-ephemeral-certs --profile profile18

# stop streaming
aria streaming stop --device-ip 192.168.8.117

# gaze detection, with mouse focus on gaze image output window, keep pressing C to calibrate while focusing on center of marker 
python3 -m projectaria_eyetracking.gaze_detect --device-ip 192.168.8.117 --device cuda:0 # or cpu

# gaze projection on egocentric image with yolo detection and gaze score calculation
python3 -m src.gaze_rgb_visualizer --device-ip 192.168.8.117 --yolo --draw-gaze --participant AB12

# record audio, press bottom right button
python3 -m src.audio_record --device-ip 192.168.8.117 --gain 2.0


# capture data for training
python3 -m src.gaze_rgb_visualizer --capture

# ros bag
python3 src/bag_record.py --participant AB12

# generate short video for aria rgb camera based on specific timestamp for arbitrary participant and trial
python3 src/clip_from_frames.py --participant AB12 --trial 04 --timestamp 1779296845648620199 --window 10
```

In ~/venv/sam3_env:
```bash
# YOLO+SAM3 service
python3 src/seg_service.py --visualize # argument for realsense segmentation image

# visualize point cloud
python3 utils/pc_viz.py 

# test command, TODO: integrate into element_action
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