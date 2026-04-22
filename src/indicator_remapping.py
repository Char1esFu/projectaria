#!/usr/bin/env python3
from evdev import InputDevice, UInput, ecodes

DEVICE = "/dev/input/by-id/usb-Wireless_Present_Wireless_Present-event-kbd"

src = InputDevice(DEVICE)

# create a virtual input device that can emit the remapped keys
ui = UInput({
    ecodes.EV_KEY: [ecodes.KEY_S, ecodes.KEY_PAGEDOWN]
}, name="presenter-remap")

# grab the source device to prevent it from sending events to other applications
src.grab()

print(f"grabbed: {src.path} {src.name}")
print("mapping: PAGEUP -> S")
print("press Ctrl+C to quit")

try:
    for event in src.read_loop():
        if event.type != ecodes.EV_KEY:
            continue

        # event.value: 1=down, 0=up, 2=repeat
        if event.code == ecodes.KEY_PAGEUP:
            ui.write(ecodes.EV_KEY, ecodes.KEY_S, event.value)
            ui.syn()
        else:
            # pass through other keys (e.g. PAGE DOWN for next slide)
            if event.code == ecodes.KEY_PAGEDOWN:
                ui.write(ecodes.EV_KEY, ecodes.KEY_PAGEDOWN, event.value)
                ui.syn()
finally:
    src.ungrab()
    ui.close()


# #!/usr/bin/env python3
# from evdev import InputDevice, ecodes

# DEVICE = "/dev/input/by-id/usb-Wireless_Present_Wireless_Present-event-kbd"

# src = InputDevice(DEVICE)
# src.grab()

# try:
#     for event in src.read_loop():
#         if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_PAGEUP:
#             print(f"code={event.code}, value={event.value}")
# finally:
#     src.ungrab()