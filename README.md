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
python3 -m projectaria_eyetracking.gaze_detect --pitch-bias 0.28 # calibrate this value by looking straight ahead and check pitch output. Compensate to 0 for each user

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

## Multi-device ROS network settings to ethernet cable connection
Install dependency:
```bash
sudo apt update
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

Verify `ip a` to check network interface name accordingly. Set up for both devices. Add cyclonedds config to `~/.ros/cyclonedds.xml`.
For workstation:
```xml
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="enp6s0"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
```
For laptop:
```xml
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="enx9cbf0d00610d"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
```

#### On workstation
```bash
# ---- Ethernet setup ----
sudo ip addr add 192.168.2.1/24 dev enp6s0 2>/dev/null
sudo ip link set enp6s0 up
sudo ip route add 224.0.0.0/4 dev enp6s0 || true

# ROS2 DDS
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=10
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
```

#### On laptop
```bash
# ---- Ethernet setup ----
sudo ip addr add 192.168.2.2/24 dev enx9cbf0d00610d 2>/dev/null
sudo ip link set enx9cbf0d00610d up
sudo ip route add 224.0.0.0/4 dev enx9cbf0d00610d || true

# ROS2 DDS
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=10
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
```

Link for reference: https://chatgpt.com/share/699763d3-2dfc-8002-b043-cbb0e3b585bb

## TODO:
Switch to WLAN connection