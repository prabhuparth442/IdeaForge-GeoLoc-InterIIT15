# Jetson Orin Nano: Use an Ubuntu Laptop as a Display

View the Jetson desktop in an mpv window through an HDMI-to-USB capture adapter while continuing to use the laptop normally. Control the Jetson with a mouse and keyboard connected directly to it.

This guide covers an already bootable Jetson Orin Nano Developer Kit with a graphical JetPack installation. It does not cover flashing JetPack. The example capture device is a **PiBox 4K video capture unit**.

> The capture adapter carries video to the laptop. It does not forward the laptop's mouse or keyboard to the Jetson. Clicking inside mpv does not control the Jetson desktop.

## 1. Requirements

| Item | Purpose |
|---|---|
| Jetson Orin Nano Developer Kit with installed OS | Runs the project and desktop |
| Suitable Jetson power supply | Powers the Jetson independently |
| Ubuntu laptop with an available USB port | Displays the video using mpv |
| PiBox HDMI-to-USB capture adapter, or equivalent | Converts the display output into a USB video feed |
| DisplayPort-to-HDMI adapter/cable and HDMI cable as needed | Connects the Jetson display output to the capture input |
| USB keyboard and mouse, or wireless USB receiver | Controls the Jetson directly |
| Internet access on the laptop during installation | Downloads mpv and diagnostic tools |

The standard NVIDIA Orin Nano Developer Kit has a **DisplayPort output**, not a native HDMI output. Use a compatible DisplayPort-to-HDMI adapter/cable. Third-party carrier boards may have different connectors.

## 2. Connect the hardware

With the Jetson powered off:

1. Connect the keyboard and mouse (or their wireless receiver) to the **Jetson USB ports**.
2. Connect the Jetson's **DisplayPort output** to the DisplayPort-to-HDMI adapter.
3. Connect the HDMI side to the capture unit's **HDMI IN** port.
4. Connect the capture unit's **USB connection to the laptop**. Prefer a direct connection rather than a hub.
5. Connect the Jetson power supply and turn it on.

If the capture unit has HDMI loop-through/output, that port is optional and is not the input from the Jetson.

The laptop's own HDMI port is normally an output; this setup receives video through USB instead.

## 3. Install the laptop software

Run these commands in the **Ubuntu laptop terminal**:

```bash
sudo apt update
sudo apt install mpv v4l-utils
```

Check installation:

```bash
mpv --version
v4l2-ctl --version
```

- **mpv** displays the capture feed.
- **v4l-utils** identifies capture devices and lists supported video modes.
- A separate FFmpeg command-line installation is not required for the mpv commands below.
- No network connection between the Jetson and laptop is required for this display setup.

## 4. Identify the Jetson capture feed

On the **laptop**, run:

```bash
v4l2-ctl --list-devices
```

Example from this setup:

```text
Acer FHD User Facing: Acer FHD:
    /dev/video0
    /dev/video1

USB3. 0 capture: USB3. 0 captur:
    /dev/video2
    /dev/video3
```

| Entry | Meaning in this example |
|---|---|
| Acer FHD User Facing | Laptop webcam |
| USB3. 0 capture | PiBox receiving the Jetson display |
| /dev/video2 | Capture video node used in the commands below |

**The device usually appears under the capture adapter's name, not “Jetson.”** The laptop sees the USB adapter; the Jetson supplies its HDMI input.

Device numbers can change after reconnecting or rebooting. Do not assume that the capture device is always `/dev/video2`.

If the names are unclear, disconnect only the capture adapter's USB connection, run the listing again, reconnect it, and compare. The disappearing/reappearing group identifies the adapter.

One physical device may expose multiple nodes, including metadata nodes. Check the candidate node with:

```bash
v4l2-ctl -d /dev/video2 --list-formats-ext
```

Use the node that lists usable **Video Capture** formats. If the command fails or lists no video modes, inspect the other node in the capture adapter's group.

## 5. Choose MJPEG to avoid the 5 FPS mode

The PiBox in this setup reported:

| Format | Resolution | Advertised frame rate |
|---|---|---|
| MJPG / MJPEG (compressed) | 1920 × 1080 | Up to 60 FPS |
| MJPG / MJPEG (compressed) | 1280 × 720 | Up to 60 FPS |
| YUYV (uncompressed) | 1920 × 1080 | 5 FPS |
| YUYV (uncompressed) | 1280 × 720 | 10 FPS |

Selecting uncompressed 1080p can explain a feed limited to 5 FPS. **Explicitly request MJPEG** instead of relying on the default format.

These are the modes advertised by this particular unit, not a guarantee for every PiBox or every “4K” capture adapter. Request only combinations shown by your device.

## 6. Open the Jetson desktop in mpv

Close any existing mpv or camera application using the capture device.

### 1080p at 60 FPS

Run on the **laptop**, replacing `/dev/video2` if necessary:

```bash
mpv av://v4l2:/dev/video2 \
  --demuxer-lavf-o=input_format=mjpeg,video_size=1920x1080,framerate=60 \
  --profile=low-latency \
  --no-audio \
  --untimed
```

### 720p at 60 FPS fallback

If 1080p stutters, try this lower-resolution mode:

```bash
mpv av://v4l2:/dev/video2 \
  --demuxer-lavf-o=input_format=mjpeg,video_size=1280x720,framerate=60 \
  --profile=low-latency \
  --no-audio \
  --untimed
```

| Option | Purpose |
|---|---|
| `av://v4l2:/dev/video2` | Opens the selected Linux capture device |
| `input_format=mjpeg` | Requests compressed Motion JPEG |
| `video_size=1920x1080` | Selects capture resolution |
| `framerate=60` | Requests 60 captured frames per second |
| `--profile=low-latency` | Applies mpv's latency-reduction settings |
| `--no-audio` | Disables audio playback |
| `--untimed` | Displays incoming video frames without normal playback timing; useful for this live, video-only feed |

The device listing calls the format `MJPG`; the mpv/FFmpeg input option uses `mjpeg`.

The requested capture rate does not guarantee 60 distinct source frames. On the **Jetson**, use its display settings to select **1920 × 1080 at 60 Hz**, if available. A slow source application can still look slow even when capture runs at 60 FPS.

If `--untimed` causes unstable playback, retry the same command without that option.

## 7. Log in and control the Jetson

Use the **keyboard and mouse attached to the Jetson** to log in and work. Keep mpv windowed to use laptop applications alongside the Jetson view.

If the Jetson is showing a text console, try **Ctrl + Alt + F2**, then **Ctrl + Alt + F1**, using the Jetson keyboard. The graphical session's terminal number depends on the installation.

If a graphical desktop is installed but its login service is stopped, log into the Jetson text console and run **on the Jetson**:

```bash
sudo systemctl start display-manager
```

If it fails, inspect the service:

```bash
systemctl status display-manager --no-pager
```

Starting a service does not install a missing desktop environment.

**Keyboard focus matters:**

- Jetson-connected keyboard: types into and controls the Jetson.
- Laptop keyboard: controls laptop applications, including mpv.
- With mpv focused on the laptop, `f` toggles fullscreen and `q` closes the player.
- Closing mpv does not shut down the Jetson.

## 8. Troubleshooting

### Feed remains near 5 FPS

1. Confirm that you opened the capture adapter, not the laptop webcam.
2. Use the full command with `input_format=mjpeg`.
3. Check mpv's terminal output for the selected codec, size, and frame rate.
4. Retry at 720p/60 FPS.
5. Plug the adapter directly into the laptop and try another USB port.
6. Confirm that the Jetson desktop itself is responsive and its display refresh rate is suitable.

The name “USB3. 0 capture” alone does not prove that the device negotiated a USB 3 connection. To inspect the actual USB topology and link speeds on the laptop:

```bash
lsusb -t
```

If `lsusb` is unavailable, install `usbutils`. A 480M link indicates USB 2 speed; 5000M indicates a 5 Gbit/s link. MJPEG can still work well over USB 2, depending on the device and mode.

### Device or resource busy

Close other camera viewers, recording apps, and earlier mpv instances using the capture device. Then retry.

### No such file or directory / wrong camera shown

Run `v4l2-ctl --list-devices` again and update the device path. The numbering may have changed.

### Permission denied

Inspect access on the laptop:

```bash
ls -l /dev/video2
id
```

If the device is owned by the `video` group and your account lacks access, add your laptop account:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in before retrying. Do not run mpv as root as the normal solution.

### Black screen / no signal

- Confirm that the Jetson has booted and is powered independently.
- Check the DisplayPort-to-HDMI adapter direction and compatibility.
- Ensure the cable enters the capture unit's HDMI **input**.
- Reseat the display connections.
- If necessary, test the Jetson with a physical monitor to separate display-output problems from capture problems.

### Unsupported mode / invalid argument

List the capture formats again and choose a supported format, size, and FPS combination. Do not force 60 FPS if your adapter only advertises 30 FPS.

### Motion is smooth but control feels delayed

Keep the low-latency settings and try 720p. Capture hardware adds its own delay, which mpv cannot eliminate completely.

## 9. Daily use

After the initial setup:

1. Connect the display capture chain and the Jetson keyboard/mouse.
2. Power on the Jetson.
3. Confirm the capture node if devices were reconnected.
4. Run the MJPEG mpv command on the laptop.
5. Log into the Jetson using its connected keyboard.
6. Work with the Jetson view alongside your laptop applications.

Shut down the Jetson through its desktop menu or run `sudo poweroff` **on the Jetson** before disconnecting its power supply.

## References

- [NVIDIA Jetson Orin Nano Developer Kit hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/hardware_layout.html)
- [mpv manual](https://mpv.io/manual/stable/)
- [FFmpeg Video4Linux2 input documentation](https://ffmpeg.org/ffmpeg-devices.html#video4linux2_002c-v4l2)

The capture-mode table is based on the PiBox device output supplied for this setup. The commands are configured from those advertised modes; actual playback performance must be checked on the connected hardware.

