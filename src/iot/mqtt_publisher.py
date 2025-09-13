# src/iot/mqtt_publisher.py
"""
Simple MQTT publisher used by realtime_infer.py to send alerts.
Default broker: broker.hivemq.com (public, for dev only).
"""

import paho.mqtt.client as mqtt
import json
import time

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "vehicle/drowsiness"

_client = mqtt.Client()
_client_connected = False

def init_client(broker=MQTT_BROKER, port=MQTT_PORT):
    global _client, _client_connected
    try:
        _client.connect(broker, port, 60)
        _client.loop_start()
        _client_connected = True
    except Exception as e:
        print("MQTT connect error:", e)
        _client_connected = False

def publish_alert(payload: dict):
    global _client_connected
    if not _client_connected:
        init_client()
    payload['ts'] = time.time()
    try:
        _client.publish(MQTT_TOPIC, json.dumps(payload))
    except Exception as e:
        print("Publish failed:", e)