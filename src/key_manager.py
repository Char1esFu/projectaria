import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from evdev import InputDevice, ecodes

DEVICE = "/dev/input/by-id/usb-Wireless_Present_Wireless_Present-event-kbd"


def main():
    rclpy.init()
    node = Node("key_manager")
    b_press_pub = node.create_publisher(Empty, "/key/b/press", 10)
    b_release_pub = node.create_publisher(Empty, "/key/b/release", 10)
    pageup_pub = node.create_publisher(Empty, "/key/pageup", 10)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    device = InputDevice(DEVICE)
    device.grab()
    print(f"key_manager: grabbed {DEVICE}")
    print("  B press/release → /key/b/press, /key/b/release")
    print("  PageUp press    → /key/pageup")

    try:
        for event in device.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            if event.code == ecodes.KEY_B:
                if event.value == 1:
                    b_press_pub.publish(Empty())
                elif event.value == 0:
                    b_release_pub.publish(Empty())
            elif event.code == ecodes.KEY_PAGEUP and event.value == 1:
                pageup_pub.publish(Empty())
    except KeyboardInterrupt:
        pass
    finally:
        device.ungrab()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
