# Wiring notes (prototype)

## Vibration motor (safe wiring)
- Connect motor positive to 5V through MOSFET drain/source arrangement.
- GPIO pin -> MOSFET gate via 1k resistor.
- MOSFET source -> GND.
- Motor negative -> MOSFET drain -> source.
- Place a diode across the motor terminals (cathode to +5V, anode to MOSFET drain) to protect against back EMF.

## Buzzer
- Active buzzer: connect between 5V and MOSFET controlled by GPIO in the same manner as motor.
- If using a passive speaker, use an amplifier board.

## Camera
- USB camera: plug into USB port on Pi.
- CSI camera: plug into CSI connector on Pi (follow Pi docs to enable).

## I2C sensors (MPU6050)
- SDA -> SDA pin on Pi
- SCL -> SCL pin on Pi
- VCC -> 3.3V
- GND -> GND
- Enable I2C from raspi-config if using Raspberry Pi

## Safety
- Test actuators with the Pi powered by bench power supply before vehicle integration.
- Do not connect to vehicle CAN or braking systems without certified hardware and approvals.