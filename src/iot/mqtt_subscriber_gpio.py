# src/iot/mqtt_subscriber_gpio.py
"""
MQTT subscriber that would normally run on Raspberry Pi to actuate GPIOs.
For local testing use --simulate to print alert actions instead of using GPIO.
"""

import paho.mqtt.client as mqtt
import json
import argparse
import time

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "vehicle/drowsiness"

def on_message_simulate(client, userdata, msg):
    payload = json.loads(msg.payload)
    print("[SIM] Received alert:", payload)
    # simulate vibrating and buzzer
    print("[SIM] Vibrate motor ON")
    time.sleep(0.6)
    print("[SIM] Vibrate motor OFF")

def run_simulate():
    client = mqtt.Client()
    client.on_message = on_message_simulate
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    print("Subscribed (simulate) to", MQTT_TOPIC)
    client.loop_forever()

def run_with_gpio():
    # Try importing RPi; if not available, fallback to simulate
    try:
        import RPi.GPIO as GPIO
    except Exception as e:
        print("RPi.GPIO not available; falling back to simulate.")
        run_simulate()
        return

    VIB_PIN = 18
    BUZZ_PIN = 23
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIB_PIN, GPIO.OUT)
    GPIO.setup(BUZZ_PIN, GPIO.OUT)

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload)
        if payload.get('alert') == 'drowsy':
            GPIO.output(VIB_PIN, GPIO.HIGH)
            GPIO.output(BUZZ_PIN, GPIO.HIGH)
            time.sleep(0.6)
            GPIO.output(VIB_PIN, GPIO.LOW)
            GPIO.output(BUZZ_PIN, GPIO.LOW)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (no GPIO)")
    args = parser.parse_args()
    if args.simulate:
        run_simulate()
    else:
        run_with_gpio()