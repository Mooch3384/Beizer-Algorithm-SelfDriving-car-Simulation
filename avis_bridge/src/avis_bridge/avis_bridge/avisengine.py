'''
@ 2025, Copyright AVIS Engine - Enhanced & Optimized for ROS2 Integration
'''

import cv2
import re
import base64
import time
import socket
import numpy as np


class Car:
    '''
    AVIS Engine Main Car class (Instance-isolated and optimized)
    '''

    def __init__(self, server="127.0.0.1", port=25001):
        self.server = server
        self.port = port
        self.steering_value = 0
        self.speed_value = 0
        self.sensor_status = 1
        self.image_mode = 1
        self.get_Speed = 1
        self.sensor_angle = 30

        self.sock = None
        self.is_connected = False

        self._data_format = "Speed:{},Steering:{},ImageStatus:{},SensorStatus:{},GetSpeed:{},SensorAngle:{}"
        self.data_str = ""
        self.updateData()

        self.image = None
        self.sensors = [1500, 1500, 1500]
        self.current_speed = 0

    def connect(self, server=None, port=None):
        if server:
            self.server = server
        if port:
            self.port = port

        # Close existing socket if open
        self.close()

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.server, self.port))
            self.is_connected = True
            return True
        except Exception as e:
            self.is_connected = False
            self.close()
            return False

    def close(self):
        self.is_connected = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def recvall(self):
        """
        Efficiently reads from the TCP socket until <EOF> indicator is found.
        Uses native byte searching instead of external KMP routines for performance.
        """
        if not self.sock:
            return None

        BUFFER_SIZE = 131072
        data = bytearray()
        try:
            while True:
                part = self.sock.recv(BUFFER_SIZE)
                if not part:
                    # Remote side closed connection
                    self.is_connected = False
                    return None
                data.extend(part)
                if b"<EOF>" in data:
                    break
            return data.decode("utf-8", errors="ignore")
        except (socket.timeout, socket.error):
            self.is_connected = False
            return None

    def setSteering(self, steering):
        self.steering_value = int(steering)
        self.image_mode = 0
        self.sensor_status = 0
        self.updateData()
        if not self.send_raw(self.data_str):
            return False
        time.sleep(0.005)
        return True

    def setSpeed(self, speed):
        self.speed_value = int(speed)
        self.image_mode = 0
        self.sensor_status = 0
        self.updateData()
        if not self.send_raw(self.data_str):
            return False
        time.sleep(0.005)
        return True

    def setSensorAngle(self, angle):
        self.sensor_angle = int(angle)
        self.image_mode = 0
        self.sensor_status = 0
        self.updateData()
        return self.send_raw(self.data_str)

    def send_raw(self, payload_str):
        if not self.sock or not self.is_connected:
            return False
        try:
            self.sock.sendall(payload_str.encode("utf-8"))
            return True
        except Exception:
            self.is_connected = False
            return False

    def getData(self):
        """
        Requests sensor, speed, and image payload from Avis Engine.
        Returns True if successful, False if network error occurs.
        """
        self.image_mode = 1
        self.sensor_status = 1
        self.updateData()

        if not self.send_raw(self.data_str):
            return False

        receive = self.recvall()
        if not receive:
            return False

        imageTagCheck = re.search(r'<image>(.*?)</image>', receive)
        sensorTagCheck = re.search(r'<sensor>(.*?)</sensor>', receive)
        speedTagCheck = re.search(r'<speed>(.*?)</speed>', receive)

        try:
            if imageTagCheck:
                imageData = imageTagCheck.group(1)
                im_bytes = base64.b64decode(imageData)
                im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
                imageOpenCV = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
                if imageOpenCV is not None:
                    self.image = imageOpenCV

            if sensorTagCheck:
                sensorData = sensorTagCheck.group(1)
                sensor_arr = re.findall(r"\d+", sensorData)
                if sensor_arr:
                    self.sensors = list(map(int, sensor_arr))
            else:
                self.sensors = [1500, 1500, 1500]

            if speedTagCheck:
                current_sp = speedTagCheck.group(1)
                if current_sp.lstrip('-').isdigit():
                    self.current_speed = int(current_sp)
            else:
                self.current_speed = 0

            return True
        except Exception:
            return False

    def getImage(self):
        return self.image

    def getSensors(self):
        return self.sensors

    def getSpeed(self):
        return self.current_speed

    def updateData(self):
        data = [
            int(self.speed_value),
            int(self.steering_value),
            int(self.image_mode),
            int(self.sensor_status),
            int(self.get_Speed),
            int(self.sensor_angle)
        ]
        self.data_str = self._data_format.format(*data)

    def stop(self):
        try:
            self.setSpeed(0)
            self.setSteering(0)
            self.send_raw("stop")
        except Exception:
            pass
        finally:
            self.close()

    def __del__(self):
        self.stop()
