# Driver Drowsiness Detection — QML + IoT (Software-first)

## Overview
--------
A from-scratch project combining classic ML/CNN, a small PennyLane hybrid prototype, and an MQTT-based IoT alert pipeline. Software-first: get the detection working on a laptop/webcam, then deploy to Raspberry Pi and add actuators.

## Structure
---------
See repository root for `src/` and `hardware/` directories.

## Quickstart (software demo)
--------------------------
1. Create & activate venv:
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the realtime demo (webcam):
```bash
python src/realtime_infer.py
```
- It opens your webcam, shows EAR/MAR values, and will publish MQTT alerts to `vehicle/drowsiness` (broker default: `broker.hivemq.com`) when drowsiness is detected.

3. Run a subscriber locally to see alerts:
```bash
python src/iot/mqtt_subscriber_gpio.py --simulate
```

## Training
--------
- See `src/train.py` for a small training pipeline. Put preprocessed eye/face patches under `data/train/alert` and `data/train/drowsy`. The dataset class expects this layout.

## Hardware
--------
- See `hardware/parts_list.md` and `hardware/wiring.md` for recommended parts and wiring instructions (safe actuators only).

## Safety & Ethics
--------------
- Do not attempt to connect to steering/braking or other vehicle-critical systems without certifications.
- Prefer on-device processing; don't stream raw video off-device unless you have explicit user consent.

If you want additional files (full notebooks, Dockerfile, or Pi image setup), ask for them specifically.