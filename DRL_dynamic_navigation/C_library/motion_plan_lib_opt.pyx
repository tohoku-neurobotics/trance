## python setup.py build_ext --inplace ####
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sin, cos, fabs, sqrt, atan2

def cal_laser(np.ndarray[np.float32_t, ndim=2] circles,
              np.ndarray[np.float32_t, ndim=1] robot_pose):

    cdef float PI = 3.141592654
    cdef float laser_resolution = 0.003490659
    cdef int num_scan = 1800
    cdef int num_circle = 7

    cdef float x0 = robot_pose[0]
    cdef float y0 = robot_pose[1]
    cdef float yaw = robot_pose[2]

    # use memoryviews for fast access
    cdef float[:, :] circles_mv = circles

    cdef np.ndarray[np.float32_t, ndim=1] scans = np.empty(num_scan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] scan_lines = np.empty((num_scan, 4), dtype=np.float32)

    # constants
    cdef float default_scan_range = 20.0
    cdef float float_num_scan = <float>num_scan
    cdef float center_index = (float_num_scan - 1.0) / 2.0

    # local vars
    cdef int i, k
    cdef float float_i, angle_rel, angle_abs, line_unit_vector_1, line_unit_vector_2
    cdef float a1, b1, c1, x2, y2, r, f2, d, a3, b3, c3, intermediate_x, intermediate_y
    cdef float temp, l_vector, intersection_x_1, intersection_y_1, intersection_x_2, intersection_y_2
    cdef float line_vector_11, line_vector_12, line_vector_21, line_vector_22
    cdef float min_range, scan_range, _intersection_x, _intersection_y

    # parallelize outer loop
    for i in prange(num_scan, nogil=True):
        float_i = <float>i
        angle_rel = (float_i - center_index) * laser_resolution

        # clamp [-PI, PI]
        if angle_rel < -PI:
            angle_rel = -PI
        elif angle_rel > PI:
            angle_rel = PI

        angle_abs = angle_rel + yaw
        line_unit_vector_1 = cos(angle_abs)
        line_unit_vector_2 = sin(angle_abs)
        min_range = default_scan_range

        _intersection_x = x0 + default_scan_range * line_unit_vector_1
        _intersection_y = y0 + default_scan_range * line_unit_vector_2

        a1 = line_unit_vector_2
        b1 = -line_unit_vector_1
        c1 = y0 * line_unit_vector_1 - x0 * line_unit_vector_2

        for k in range(num_circle):
            if circles_mv[k, 0] < -50.0:
                break

            x2 = circles_mv[k, 0]
            y2 = circles_mv[k, 1]
            r = circles_mv[k, 2]

            f2 = a1 * x2 + b1 * y2 + c1
            d = fabs(f2)

            if d > r:
                continue

            # compute intermediate line
            a3 = b1
            b3 = -a1
            c3 = -(a3 * x2 + b3 * y2)
            intermediate_x = (b1 * c3 - b3 * c1) / (a1 * b3 - a3 * b1)
            intermediate_y = (a3 * c1 - a1 * c3) / (a1 * b3 - a3 * b1)

            if d == r:
                _intersection_x = intermediate_x
                _intersection_y = intermediate_y
                min_range = sqrt((_intersection_x - x0)**2 + (_intersection_y - y0)**2)
            else:
                temp = r * r - d * d
                if temp <= 0.0:
                    continue

                l_vector = sqrt(temp)

                intersection_x_1 = intermediate_x + l_vector * line_unit_vector_1
                intersection_y_1 = intermediate_y + l_vector * line_unit_vector_2
                intersection_x_2 = intermediate_x - l_vector * line_unit_vector_1
                intersection_y_2 = intermediate_y - l_vector * line_unit_vector_2

                # first intersection
                line_vector_11 = intersection_x_1 - x0
                line_vector_12 = intersection_y_1 - y0
                if line_vector_11 * line_unit_vector_1 > 0.0:
                    scan_range = sqrt(line_vector_11 * line_vector_11 + line_vector_12 * line_vector_12)
                    if scan_range < min_range:
                        min_range = scan_range
                        _intersection_x = intersection_x_1
                        _intersection_y = intersection_y_1

                # second intersection
                line_vector_21 = intersection_x_2 - x0
                line_vector_22 = intersection_y_2 - y0
                if line_vector_21 * line_unit_vector_1 > 0.0:
                    scan_range = sqrt(line_vector_21 * line_vector_21 + line_vector_22 * line_vector_22)
                    if scan_range < min_range:
                        min_range = scan_range
                        _intersection_x = intersection_x_2
                        _intersection_y = intersection_y_2

        scans[i] = min_range
        scan_lines[i, 0] = x0
        scan_lines[i, 1] = y0
        scan_lines[i, 2] = _intersection_x
        scan_lines[i, 3] = _intersection_y

    return scans, scan_lines

def transform_scan_last(np.ndarray[np.float32_t, ndim=3] scan_end_last,
                        np.ndarray[np.float32_t, ndim=1] robot_pose):
    cdef float PI = 3.141592654
    cdef float laser_resolution = 0.003490659
    cdef int num_scan = 1800
    cdef int num_layers = 5

    cdef float x0 = robot_pose[0]
    cdef float y0 = robot_pose[1]
    cdef float yaw = robot_pose[2]
    cdef float s_yaw = sin(yaw)
    cdef float c_yaw = cos(yaw)

    cdef np.ndarray[np.float32_t, ndim=2] scan_last = np.full((num_layers, num_scan), 20.0, dtype=np.float32)

    cdef float dx, dy, scan_x, scan_y, scan_angle
    cdef int i, layer, scan_index

    cdef float center_index = PI - laser_resolution / 2.0

    for layer in range(num_layers):
        for i in range(num_scan):
            dx = scan_end_last[layer, i, 0] - x0
            dy = scan_end_last[layer, i, 1] - y0

            # rotation
            scan_x = dy * s_yaw + dx * c_yaw
            scan_y = dy * c_yaw - dx * s_yaw

            # compute angle and index
            scan_angle = atan2(scan_y, scan_x)
            if -PI <= scan_angle <= PI:
                scan_index = <int>((scan_angle + center_index) / laser_resolution)
                if 0 <= scan_index < num_scan:
                    scan_last[layer, scan_index] = sqrt(scan_x * scan_x + scan_y * scan_y)

    return scan_last
