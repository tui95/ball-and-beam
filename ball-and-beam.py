from collections import deque
import time
import warnings
import math
import serial
import serial.tools.list_ports
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream


def connect_arduino(baudrate=9600):  # a more civilized way to connect to arduino
    def is_arduino(p):
        # need more comprehensive test
        return p.manufacturer is not None and 'arduino' in p.manufacturer.lower()

    ports = serial.tools.list_ports.comports()
    arduino_ports = [p for p in ports if is_arduino(p)]

    def port2str(p):
        return "%s - %s (%s)" % (p.device, p.description, p.manufacturer)

    if not arduino_ports:
        portlist = "\n".join([port2str(p) for p in ports])
        raise IOError("No Arduino found\n" + portlist)

    if len(arduino_ports) > 1:
        portlist = "\n".join([port2str(p) for p in ports])
        warnings.warn('Multiple Arduinos found - using the first\n' + portlist)

    selected_port = arduino_ports[0]
    print("Using %s" % port2str(selected_port))
    ser = serial.Serial(selected_port.device, baudrate)
    time.sleep(2)  # this is important it takes time to handshake
    return ser


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# ORANGE_LOWER = (0, 64, 255)
ORANGE_LOWER = (0, 24, 95)
ORANGE_UPPER = (23, 255, 181)

BALANCE_ANGLE = 80  # servo angle
INTERVAL = 200


class ArduinoConnector:
    def __init__(self, ser):
        self.ser = ser

    def send_rec(self, msg):
        self.ser.write((msg + "\n").encode())

class BallAndBeam:
    def __init__(self, x_center, y_center):
        self.x_center = x_center
        self.y_center = y_center
        self.h = 107-y_center

    def compute_error(self, point):
        x, y = point
        x_diff = self.x_center - x
        y_diff = self.y_center - y

        sign = -1 if x < self.x_center else 1

        euclidian_dist = np.sqrt(x_diff**2 + y_diff**2)
        theta = np.arcsin(self.h/euclidian_dist)
        if abs(self.h/euclidian_dist) > 0.9999999:
            theta = math.pi/2

        distance_from_center = euclidian_dist*np.cos(theta)
        distance_from_center *= sign
        return distance_from_center


class PID:
    def __init__(self, ball_and_beam):
        self.ball_and_beam = ball_and_beam
        base_k = 6/115
        self.kp = base_k/6
        self.ki = base_k/24
        self.kd = 6

        ndata = 5
        self.data = deque([], ndata)
        self.integral = 0
        self.derivative = 0

    def compute_angle(self, error):
        p = self.compute_proportional(error)
        i = self.compute_integral()
        d = self.compute_derivative()
        angle =  self.add_balance_angle(self.kp*p + self.ki*i + self.kd*d)
        return angle

    def add_balance_angle(self, alpha):
        alpha += BALANCE_ANGLE
        alpha = int(round(alpha))
        return alpha

    def compute_proportional(self, error):
        return error

    def compute_integral(self):
        integral = sum(self.data)/len(self.data)
        self.integral = integral
        return integral

    def compute_derivative(self):
        if len(self.data) < 5:
            self.derivative = 0
            return self.derivative
        var = np.var(self.data)
        if var < 1:
            self.derivative = 0
            return self.derivative
        y = np.arange(len(self.data))
        exy = np.mean(self.data*y)
        ex = np.mean(self.data)
        ey = np.mean(y)
        cov = exy-ex*ey

        self.derivative = cov/var
        if np.isnan(self.derivative):
            self.derivative = 0
        return self.derivative

def millis():
    return int(round(time.time() * 1000))

def display_txt(frame, txt, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.6
    line_colour = (0, 0, 255)
    thickness = 2
    cv2.putText(frame, txt, position, font, font_size, line_colour, thickness, cv2.LINE_4)

def main():
    buffer_size = 64
    pts = deque(maxlen=buffer_size)
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)

    with connect_arduino() as ser:
        arduino_connector = ArduinoConnector(ser)
        arduino_connector.send_rec(str(BALANCE_ANGLE))
        ball_and_beam = BallAndBeam(x_center=260/2, y_center=334/2)
        pid = PID(ball_and_beam)

        # keep looping
        while True:
            # grab the current frame
            _, frame = cap.read()
            prev_time = millis()

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if frame is None:
                break

            # resize the frame, blur it, and convert it to the HSV
            # color space
            # frame = imutils.resize(frame, width=600)
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # calculate error, pid
                error = ball_and_beam.compute_error(center)
                pid.data.append(error)

                servo_angle = pid.compute_angle(error)

                arduino_connector.send_rec(str(servo_angle))


                display_txt(frame, f'p: {error:.2f}', (10, 20))
                display_txt(frame, f'i: {pid.integral:.2f}', (10, 50))
                display_txt(frame, f'd: {pid.derivative:.2f}', (10, 80))
                display_txt(frame, f'kp*p: {pid.kp*error:.2f}', (200, 20))
                display_txt(frame, f'ki*i: {pid.ki*pid.integral:.2f}', (200, 50))
                display_txt(frame, f'kd*d: {pid.kd*pid.derivative:.2f}', (200, 80))
                display_txt(frame, f'servo_angle: {servo_angle}', (10, 220))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)

            # show the frame to our screen
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

        # stop the camera video stream
        cap.stop()
        # close all windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
