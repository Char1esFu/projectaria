import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from evdev import InputDevice, ecodes

PRESENTER_DEVICE = "/dev/input/by-id/usb-Wireless_Present_Wireless_Present-event-kbd"
KEYBOARD_DEVICE = "/dev/input/by-id/usb-_HP_310_Wired_Keyboard-event-kbd"


def _watch_presenter(device, b_press_pub, b_release_pub, stop_event):
    try:
        for event in device.read_loop():
            if stop_event.is_set():
                break
            if event.type != ecodes.EV_KEY:
                continue
            if event.code == ecodes.KEY_B:
                if event.value == 1:
                    b_press_pub.publish(Empty())
                elif event.value == 0:
                    b_release_pub.publish(Empty())
    except Exception:
        pass


def _watch_keyboard(device, lb_pub, stop_event):
    try:
        for event in device.read_loop():
            if stop_event.is_set():
                break
            if event.type != ecodes.EV_KEY:
                continue
            if event.code == ecodes.KEY_LEFTBRACE and event.value == 1:
                lb_pub.publish(Empty())
    except Exception:
        pass


def main():
    rclpy.init()
    node = Node("key_manager")
    b_press_pub = node.create_publisher(Empty, "/key/b/press", 10)
    b_release_pub = node.create_publisher(Empty, "/key/b/release", 10)
    lb_pub = node.create_publisher(Empty, "/key/leftbrace", 10)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    presenter = InputDevice(PRESENTER_DEVICE)
    keyboard = InputDevice(KEYBOARD_DEVICE)
    presenter.grab()
    # keyboard.grab()
    print(f"key_manager: grabbed {PRESENTER_DEVICE}")
    print(f"key_manager: grabbed {KEYBOARD_DEVICE}")
    print("  B press/release → /key/b/press, /key/b/release")
    print("  [ press         → /key/leftbrace  (starts calibration; auto-stops after --calib-duration seconds)")

    stop_event = threading.Event()
    t1 = threading.Thread(target=_watch_presenter, args=(presenter, b_press_pub, b_release_pub, stop_event), daemon=True)
    t2 = threading.Thread(target=_watch_keyboard, args=(keyboard, lb_pub, stop_event), daemon=True)
    t1.start()
    t2.start()

    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        presenter.ungrab()
        keyboard.ungrab()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
