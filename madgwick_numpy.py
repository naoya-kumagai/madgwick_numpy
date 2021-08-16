# -*- coding: utf-8 -*-

import warnings
import numpy as np
from numpy.linalg import norm
import quaternion
from quaternion import as_quat_array


class MadgwickAHRS:
    
    quaternion = np.quaternion(1, 0, 0, 0)
    beta = np.sqrt(3/4)*np.pi*(5.0/180.0)

    def __init__(self, quaternion=None, beta=None):

        """
        Initialize the class with the given parameters.
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :return:
        """

        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param dt: Time elaspsed since last update. 
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        acc_norm = accelerometer / norm(accelerometer)

        # Normalise magnetometer measurement
        if norm(magnetometer) == 0:
            warnings.warn("magnetometer is zero")
            return
        mag_norm = magnetometer / norm(magnetometer)

        h = q * (np.quaternion(0, mag_norm[0], mag_norm[1], mag_norm[2]) * q.conj())
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q.x*q.z - q.w*q.y) - acc_norm[0],
            2*(q.w*q.x + q.y*q.z) - acc_norm[1],
            2*(0.5 - q.x**2 - q.y**2) - acc_norm[2],
            2*b[1]*(0.5 - q.y**2 - q.z**2) + 2*b[3]*(q.x*q.z - q.w*q.y) - mag_norm[0],
            2*b[1]*(q.x*q.y - q.w*q.z) + 2*b[3]*(q.w*q.x + q.y*q.z) - mag_norm[1],
            2*b[1]*(q.w*q.y + q.x*q.z) + 2*b[3]*(0.5 - q.x**2 - q.y**2) - mag_norm[2]
        ])
        j = np.array([
            [-2*q.y,                  2*q.z,                  -2*q.w,                  2*q.x],
            [2*q.x,                   2*q.w,                  2*q.z,                   2*q.y],
            [0,                        -4*q.x,                 -4*q.y,                  0],
            [-2*b[3]*q.y,             2*b[3]*q.z,             -4*b[1]*q.y-2*b[3]*q.w, -4*b[1]*q.z+2*b[3]*q.x],
            [-2*b[1]*q.z+2*b[3]*q.x, 2*b[1]*q.y+2*b[3]*q.w, 2*b[1]*q.x+2*b[3]*q.z,  -2*b[1]*q.w+2*b[3]*q.y],
            [2*b[1]*q.y,              2*b[1]*q.z-4*b[3]*q.x, 2*b[1]*q.w-4*b[3]*q.y,  2*b[1]*q.x]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * np.quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * as_quat_array(step.T)

        # Integrate to yield quaternion
        q += qdot * dt
        
        self.quaternion = q / q.norm()  # normalise quaternion

    def update_imu(self, dt, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param dt: Time elaspsed since last update. 
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        acc_norm = accelerometer / norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q.x*q.z - q.w*q.y) - acc_norm[0],
            2*(q.w*q.x + q.y*q.z) - acc_norm[1],
            2*(0.5 - q.x**2 - q.y**2) - acc_norm[2]
        ])
        j = np.array([
            [-2*q.y, 2*q.z, -2*q.w, 2*q.x],
            [2*q.x, 2*q.w, 2*q.z, 2*q.y],
            [0, -4*q.x, -4*q.y, 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * np.quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * as_quat_array(step.T)

        # Integrate to yield quaternion
        q += qdot * dt

        self.quaternion = q / q.norm()  # normalise quaternion
