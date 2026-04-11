import time
import argparse
import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped



FINGER_COLORS = [
    (0, 0, 255),    # thumb - red
    (0, 165, 255),  # index - orange
    (0, 255, 0),    # middle - green
    (255, 0, 0),    # ring - blue
    (255, 0, 255),  # pinky - magenta
]

FINGER_EDGES = [
    [(0,1),(1,2),(2,3),(3,4)],
    [(0,5),(5,6),(6,7),(7,8)],
    [(0,9),(9,10),(10,11),(11,12)],
    [(0,13),(13,14),(14,15),(15,16)],
    [(0,17),(17,18),(18,19),(19,20)],
]

# Finger colors for RViz (RGBA, 0-1 float) matching FINGER_COLORS
FINGER_COLORS_RGBA = [
    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),   # thumb - red
    ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0),   # index - orange
    ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),    # middle - green
    ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),    # ring - blue
    ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),    # pinky - magenta
]

# Map each joint to its finger index (0-4)
JOINT_TO_FINGER = [
    -1,        # 0: wrist
    0,0,0,0,   # 1-4: thumb
    1,1,1,1,   # 5-8: index
    2,2,2,2,   # 9-12: middle
    3,3,3,3,   # 13-16: ring
    4,4,4,4,   # 17-20: pinky
]

def draw_hand_skeleton(img, kpts_2d, is_right):
    """Draw 2D hand skeleton on image."""
    for finger_idx, edges in enumerate(FINGER_EDGES):
        color = FINGER_COLORS[finger_idx]
        for (i, j) in edges:
            pt1 = (int(kpts_2d[i, 0]), int(kpts_2d[i, 1]))
            pt2 = (int(kpts_2d[j, 0]), int(kpts_2d[j, 1]))
            cv2.line(img, pt1, pt2, color, 2)

    for k in range(21):
        pt = (int(kpts_2d[k, 0]), int(kpts_2d[k, 1]))
        cv2.circle(img, pt, 3, (255, 255, 255), -1)
        cv2.circle(img, pt, 2, (0, 0, 0), -1)

    # Label R/L near wrist
    wrist = (int(kpts_2d[0, 0]) + 10, int(kpts_2d[0, 1]))
    label = "R" if is_right > 0.5 else "L"
    cv2.putText(img, label, wrist, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def build_hand_markers(hand_idx, kpts_3d, cam_t, is_right, stamp, frame_id="camera_optical_link"):
    """Build MarkerArray for one hand: joints (SPHERE_LIST) + bones (LINE_LIST)."""
    markers = []
    # Transform to camera space
    # Note: pipeline already flips x-axis of pred_keypoints_3d for left hands,
    # and pred_cam_t_full is computed in the corrected frame. No additional flip needed.
    pts = kpts_3d.copy() + cam_t

    ns = f"hand_{hand_idx}"
    header = Header(stamp=stamp, frame_id=frame_id)

    # --- Joint spheres ---
    joint_marker = Marker()
    joint_marker.header = header
    joint_marker.ns = ns
    joint_marker.id = hand_idx * 10
    joint_marker.type = Marker.SPHERE_LIST
    joint_marker.action = Marker.ADD
    joint_marker.scale.x = 0.008
    joint_marker.scale.y = 0.008
    joint_marker.scale.z = 0.008
    joint_marker.pose.orientation.w = 1.0

    for k in range(21):
        p = Point(x=float(pts[k, 0]), y=float(pts[k, 1]), z=float(pts[k, 2]))
        joint_marker.points.append(p)
        fi = JOINT_TO_FINGER[k]
        joint_marker.colors.append(FINGER_COLORS_RGBA[fi])

    markers.append(joint_marker)

    # --- Bone lines ---
    line_marker = Marker()
    line_marker.header = header
    line_marker.ns = ns
    line_marker.id = hand_idx * 10 + 1
    line_marker.type = Marker.LINE_LIST
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.004
    line_marker.pose.orientation.w = 1.0

    for fi, edges in enumerate(FINGER_EDGES):
        color = FINGER_COLORS_RGBA[fi]
        for (i, j) in edges:
            line_marker.points.append(Point(x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2])))
            line_marker.points.append(Point(x=float(pts[j, 0]), y=float(pts[j, 1]), z=float(pts[j, 2])))
            line_marker.colors.append(color)
            line_marker.colors.append(color)

    markers.append(line_marker)

    # --- Label text ---
    text_marker = Marker()
    text_marker.header = header
    text_marker.ns = ns
    text_marker.id = hand_idx * 10 + 2
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position = Point(x=float(pts[0, 0]), y=float(pts[0, 1]) - 0.03, z=float(pts[0, 2]))
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.02
    text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    text_marker.text = "R" if is_right > 0.5 else "L"
    markers.append(text_marker)

    return markers


def main():
    parser = argparse.ArgumentParser(description='WiLoR-Mini webcam demo')
    parser.add_argument('--cam_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--focal_length', type=float, default=None,
                        help='Actual camera focal length in pixels at native resolution. '
                             'If not set, defaults to 300 * max(W,H) / 256 (WiLoR internal default).')
    args = parser.parse_args()

    # Open camera first to get resolution for focal length scaling
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.cam_id}")
        return
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_side = max(img_w, img_h)

    # TODO: finalize focal length reading with Aria glass.
    # WiLoR expects focal_length pre-scaled to 256px space: f_wilor = f_px * 256 / max(W, H)
    if args.focal_length is not None:
        focal_length_wilor = args.focal_length * 256.0 / max_side
    else:
        focal_length_wilor = 298.9  # WiLoR default (already in 256px space)
    print(f"Camera resolution: {img_w}x{img_h}, focal_length_wilor={focal_length_wilor:.2f}")

    # Init ROS2
    rclpy.init()
    node = Node('wilor_hand_publisher')
    marker_pub = node.create_publisher(MarkerArray, '/hand_markers', 10)

    # Publish static TF tree: base_link -> camera_link -> camera_optical_link
    tf_broadcaster = StaticTransformBroadcaster(node)
    stamp = node.get_clock().now().to_msg()

    # base_link -> camera_link: z offset 0.5m, no rotation
    t1 = TransformStamped()
    t1.header.stamp = stamp
    t1.header.frame_id = 'base_link'
    t1.child_frame_id = 'camera_link'
    t1.transform.translation.z = 0.5
    t1.transform.rotation.w = 1.0

    # camera_link -> camera_optical_link: standard ROS optical frame rotation
    # camera_link: x-forward, y-left, z-up (ROS convention)
    # optical:     x-right, y-down, z-forward
    # Rotation: Ry(+90°) then Rz(-90°)
    t2 = TransformStamped()
    t2.header.stamp = stamp
    t2.header.frame_id = 'camera_link'
    t2.child_frame_id = 'camera_optical_link'
    t2.transform.rotation.w = 0.5
    t2.transform.rotation.x = -0.5
    t2.transform.rotation.y = 0.5
    t2.transform.rotation.z = -0.5

    tf_broadcaster.sendTransform([t1, t2])
    node.get_logger().info('Publishing hand markers on /hand_markers')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    print("Loading WiLoR-Mini pipeline...")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, focal_length=focal_length_wilor)
    print(f"Pipeline loaded.")

    print("Webcam started. Press 'q' to quit.")
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs = pipe.predict(image_rgb)

            display = frame.copy()
            stamp = node.get_clock().now().to_msg()
            marker_array = MarkerArray()

            # Clear previous markers
            clear_marker = Marker()
            clear_marker.header.stamp = stamp
            clear_marker.header.frame_id = "camera_optical_link"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)

            for hand_idx, hand in enumerate(outputs):
                preds = hand["wilor_preds"]
                kpts_2d = preds["pred_keypoints_2d"][0]  # (21, 2)
                kpts_3d = preds["pred_keypoints_3d"][0]  # (21, 3)
                cam_t = preds["pred_cam_t_full"][0]      # (3,)
                draw_hand_skeleton(display, kpts_2d, hand["is_right"])
                markers = build_hand_markers(hand_idx, kpts_3d, cam_t, hand["is_right"], stamp)
                marker_array.markers.extend(markers)

            marker_pub.publish(marker_array)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(display, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('WiLoR-Mini Webcam', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
