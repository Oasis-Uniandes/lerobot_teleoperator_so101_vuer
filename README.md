# SO101 Vuer VR Teleoperator

A custom teleoperation plugin for the [LeRobot](https://github.com/huggingface/lerobot) framework. This package spins up a local [Vuer](https://docs.vuer.ai/) server to stream real-time 6-DoF hand tracking and pinch gestures directly from a VR headset into your robotics pipeline.

It uses [Pyroki](https://github.com/chungmin99/pyroki) to solve Inverse Kinematics (IK) on the fly, translating your physical hand movements into joint angles for the robot arm. **It has been explicitly tested using the Meta Quest 3** via the native WebXR browser API.

## Installation

This plugin requires a working installation of LeRobot and the Pyroki IK solver.

**1. Clone the repository**
```bash
git clone https://github.com/SinfonIAUniandes/lerobot_teleoperator_so101_vuer
cd lerobot_teleoperator_so101_vuer

```

**2. Install the Vuer and IK dependencies**
This lightweight requirements file installs the web UI, WebXR server, and URDF loaders without conflicting with your core LeRobot environment:

```bash
pip install -r requirements.txt

```

**3. Install the plugin**
Install the package in editable mode so it registers with the LeRobot CLI:

```bash
pip install -e .

```

## SSL Certificate Setup (Required for WebXR)

The Meta Quest browser enforces strict security policies. WebXR APIs (which provide hand tracking) **will only run in a secure HTTPS context**. You must generate self-signed certificates to run this teleoperator.

Generate them in your working directory using OpenSSL:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem

```

## Usage

Once installed, the teleoperator is automatically discovered by the LeRobot CLI and can be referenced using `--teleop.type=so101_vuer`.

### 1. Handling Certificates and Running the Server

**Case A: Running from the same directory as your certificates**
If your terminal is in the same folder where `cert.pem` and `key.pem` are located, you can run the command normally. The plugin defaults to looking in the current working directory (`./cert.pem` and `./key.pem`).

```bash
lerobot-teleoperate \
  --robot.type=so101_mujoco \
  --teleop.type=so101_vuer

```

**Case B: Running from a different directory**
If you are executing the LeRobot command from a different folder, you must explicitly pass the absolute paths to your certificates using CLI overrides:

```bash
lerobot-teleoperate \
  --robot.type=so101_mujoco \
  --teleop.type=so101_vuer \
  --teleop.vuer_cert=/absolute/path/to/cert.pem \
  --teleop.vuer_key=/absolute/path/to/key.pem

```

*(Note: You can also configure the tracked hand, coordinate system, and user height via CLI overrides like `--teleop.user_hand=left` or `--teleop.user_height=1.75`).*

### 2. Connecting the Meta Quest 3

Once the LeRobot command is running and the Vuer server has started:

1. Ensure your PC and your Meta Quest 3 are connected to the **exact same Wi-Fi network**.
2. Find your PC's local IP address (e.g., `192.168.1.50`).
3. Put on the Quest 3, open the **Meta Quest Browser**, and navigate to the following URL (replacing `<YOUR_IP>` with your actual IP address):
```text
https://<YOUR_IP>:8012/?ws=wss://<YOUR_IP>:8012

```


*(You must explicitly type both `https://` and the `?ws=wss://` parameters).*
4. The browser will warn you that the connection is not private. Click **Advanced** and proceed anyway.
5. Stand up straight, look forward, and **recenter your Quest view** (hold the Meta button or pinch the menu icon). This calibration is required for the static torso mathematics to align your virtual shoulder with the robot's base.
6. Click the **Enter VR** button at the bottom of the webpage. Your hand movements will now drive the robot!

### Recording Data

To collect demonstrations for training a model:

```bash
lerobot-record \
  --robot.type=so101_mujoco \
  --teleop.type=so101_vuer \
  --dataset.repo_id=local/vuer_vr_test \
  --dataset.single_task="pick up the object"

```
