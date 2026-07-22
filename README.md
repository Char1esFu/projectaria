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
aria streaming start --interface wifi --device-ip 192.168.8.117 --use-ephemeral-certs --profile profile18
aria streaming start --interface usb --use-ephemeral-certs --profile profile5 # profile5 20FPS for both RGB & eyetrack and use usb connection
# stop streaming
aria streaming stop --device-ip 192.168.8.117
```
In .venv:
```bash
# indicator key mapping, PAGEUP & B grabbed by evdev
python3 src/key_manager.py

# gaze detection, with mouse focus on gaze image output window, press top left button on the indicator to start collecting calibration data while focusing on center of marker, press once more to stop.
python3 -m projectaria_eyetracking.gaze_detect

# gaze projection on egocentric image with yolo detection and gaze score calculation.
# --rgb-buffer-delay-frames: delay display/recording to sync frames with gaze.
# --rgb-timestamp-source {zedr,hardware}: stamp recorded frames with the ZED right camera_info stamp or each Aria RGB frame's hardware capture time.
# --gaze-select-method {msd,variance}: stable-frame selector feeding the /gaze_label average. 'msd' is the original pixel-space dense-point method on the stitched gaze track; 'variance' selects low label-score-variance windows and averages only each selected window's centre frame.
# variance-method knobs: --gaze-var-window (frames per variance window), --gaze-var-threshold (max variance), --gaze-var-top (keep N lowest, <=0 = all), --gaze-var-force-endpoint-points (points force-excluded at a non-fixated endpoint). Boundary radius reuses --gaze-peak-radius. On stop it also writes stitched_variance.png and a variance_selected/ subfolder (selected frames' images + scores.json).
python3 -m src.gaze_rgb_visualizer --yolo --draw-gaze --participant test02 --rgb-buffer-delay-frames 5 --gaze-select-method variance --gaze-var-window 5 --gaze-var-threshold 50 --gaze-var-top 5 --gaze-var-force-endpoint-points 3 --rgb-timestamp-source hardware

# offline variance point-selection on a saved recording (or pass an experiment folder like recordings/test02 to batch every take).
# -w window, -t variance threshold, -n keep N lowest (omit = all under threshold), --force-endpoint-points force-excluded endpoint points. Writes gaze_score_stability_variance.png (variance-vs-time), stitched_variance.png, a frames grid, and variance_selected/. --no-plot / --no-export to skip.
python3 src/gaze_score_stability.py recordings/test02/33 -w 3 -t 50 -n 5 --force-endpoint-points 3

# plot per-label detection score over time (one line per label) into the run's gaze_scores.png; pass an experiment folder (test02) to do every sub-run
python src/plot_gaze_scores.py test02/32

# report per-point local MSD of the stitched gaze track and highlight the low-MSD selected points; accepts one track file or an experiment directory
python3 -m src.gaze_peak_window_stats recordings/test02/32

# record audio, long press bottom right button, if --source local is added, use PulseAudio's default microphone
python3 -m src.audio_record --participant AB12 --source local
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
python3 utils/pcd_viz.py 

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
Local network time alignment across devices:
```bash
# install on both server and client computer
sudo apt install chrony

# edit config file
sudo nano /etc/chrony/chrony.conf

# on main workstation, add following to the conf
allow 192.168.8.x/24
local stratum 10

sudo systemctl restart chrony.service

# on external computer, add following to the corresponding conf
# pool ntp.ubuntu.com iburst # comment out this line
server 192.168.8.xxx iburst

sudo systemctl restart chrony.service
sudo chronyc makestep # skip timesteps and align immediately

# verify
chronyc tracking
chronyc source -v
```

#### Setup DDS config on workstation and laptop
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=10
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
```