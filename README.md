## Installation guide
```bash
uv venv .venv -p 3.10
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install projectaria_client_sdk==1.1.0 --no-cache-dir --prerelease=allow
```

When the glass is reset or paired to a new device, setup everything in the companion app first. Then follow the instructions below. 
## Initialization
```bash
aria-doctor

aria auth pair # check phone notification to confirm pairing
aria streaming install-certs
```

## Usage
Scenario 1: Connect the glass to workstation --> usb
Scenario 2: Connect the glass to a laptop, publish topics to the workstation on local network through ros --> usb
Scenario 3: Publish sensor data on the glass directly through local network(avoid public network due to firewalls) --> wifi
```bash
# streaming demo with glfw window. Run this first to give sudo access, otherwise streaming will fail later.
python3 utils/device_stream.py --interface usb --update_iptables
# or 
python3 utils/device_stream.py --interface wifi --device-ip 192.168.8.117 --update_iptables

# subscribe first(needed in all scenarios)
aria streaming start --interface usb --use-ephemeral-certs --profile profile23 # this profile supports high rgb fps
# or 
aria streaming start --interface wifi --device-ip 192.168.8.117 --use-ephemeral-certs --profile profile23

# glass localization, specify arbitrary aruco marker id(s) in the setup
source /opt/ros/humble/setup.bash
python3 -m src.aruco_localization --device-ip 192.168.8.117 --marker-ids 0 1 2 # list all used marker ids in argument

# gaze detection
# with mouse focus on gaze image output window, keep pressing C to calibrate while focusing on center of marker 
python3 -m projectaria_eyetracking.gaze_detect --device cuda:0

# publish gaze table intersection
python3 src/gaze_intersection_node.py --visualize

# record audio
python -m src.audio_record --update_iptables --channel 0

# hand gesture detection on egocentric camera
python3 -m src.hand_gesture --device-ip 192.168.8.117

# gaze projection on egocentric image
python -m src.gaze_rgb_visualizer --device-ip 192.168.8.117
```

## Utils
```bash
# estimate pose with IMU data and Madgwick filter as failsafe
python3 -m src.imu_pose --imu-index 1 # or 0

# streaming demo with individual message
python3 -m utils.streaming_subscribe
aria streaming stop --device-ip 192.168.8.117
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