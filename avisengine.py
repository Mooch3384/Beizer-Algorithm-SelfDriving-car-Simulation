#!/usr/bin/env python3
"""
Official Driver & Client API for AVIS Engine Simulation.
Handles socket communication, data parsing, telemetry, sensors, and frame decoding.
"""
import socket
import base64
import numpy as np
import cv2
import time
import re


class Car:
    def __init__(self):
        self.sock = None
        self.connected = False
        self.image = None
        self.speed = 0.0
        self.sensors = [100.0, 100.0, 100.0]  # [left, middle, right] in cm
        self.buffer = ""

    def connect(self, ip="127.0.0.1", port=25001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((ip, int(port)))
        self.connected = True

    def setSpeed(self, speed):
        """Set vehicle throttle (-100 to 100)"""
        if self.connected and self.sock:
            try:
                cmd = f"<speed>{int(speed)}</speed>\n"
                self.sock.sendall(cmd.encode('utf-8'))
            except Exception:
                pass

    def setSteering(self, steering):
        """Set vehicle steering angle (-100 to 100)"""
        if self.connected and self.sock:
            try:
                cmd = f"<steering>{int(steering)}</steering>\n"
                self.sock.sendall(cmd.encode('utf-8'))
            except Exception:
                pass

    def setSensorAngle(self, angle):
        """Configure distance sensors field of view angle"""
        if self.connected and self.sock:
            try:
                cmd = f"<sensorAngle>{int(angle)}</sensorAngle>\n"
                self.sock.sendall(cmd.encode('utf-8'))
            except Exception:
                pass

    def getData(self):
        """Retrieve and decode latest frames and sensor packets from simulator"""
        if not self.connected or not self.sock:
            return

        try:
            self.sock.settimeout(0.05)
            while True:
                data = self.sock.recv(16384)
                if not data:
                    break
                self.buffer += data.decode('utf-8', errors='ignore')
                if '</EOF>' in self.buffer or '\n' in self.buffer:
                    break
        except socket.timeout:
            pass
        except Exception:
            return

        # Split and process complete messages
        while '</EOF>' in self.buffer or '\n' in self.buffer:
            if '</EOF>' in self.buffer:
                parts = self.buffer.split('</EOF>', 1)
            else:
                parts = self.buffer.split('\n', 1)
            msg = parts[0]
            self.buffer = parts[1]
            self._parse_payload(msg)

    def _parse_payload(self, msg):
        # 1. Parse Image Frame
        if '<image>' in msg and '</image>' in msg:
            try:
                img_str = msg.split('<image>')[1].split('</image>')[0]
                img_bytes = base64.b64decode(img_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    self.image = decoded
            except Exception:
                pass

        # 2. Parse Speed Telemetry
        if '<speed>' in msg and '</speed>' in msg:
            try:
                s_str = msg.split('<speed>')[1].split('</speed>')[0]
                self.speed = float(s_str)
            except Exception:
                pass

        # 3. Parse Distance Sensors (left, middle, right)
        if '<sensors>' in msg and '</sensors>' in msg:
            try:
                sens_str = msg.split('<sensors>')[1].split('</sensors>')[0]
                vals = [float(x) for x in sens_str.split(',') if x.strip()]
                if len(vals) >= 3:
                    self.sensors = vals[:3]
            except Exception:
                pass

    def getImage(self):
        """Returns camera image as OpenCV compatible numpy array"""
        return self.image

    def getSpeed(self):
        """Returns current vehicle speed in km/h"""
        return self.speed

    def getSensors(self):
        """Returns distance sensor readings [left, middle, right] in cm"""
        return self.sensors

    def stop(self):
        """Safely stops vehicle and releases connection"""
        if self.connected and self.sock:
            try:
                self.setSpeed(0)
                self.setSteering(0)
                self.sock.close()
            except Exception:
                pass
        self.connected = False