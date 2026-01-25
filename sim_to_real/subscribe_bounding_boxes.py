#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from threading import Lock
from sklearn.cluster import DBSCAN
from math import fabs, hypot, cos, sin, atan2

class SubBoundingBox:
    def __init__(self):        
        self.sub_bounding_boxes = rospy.Subscriber("/bounding_boxes", Float32MultiArray, 
                                                   self.callback_bounding_boxes)
        self.sub_robot_pose_2d = rospy.Subscriber("/robot/pose2d", Float32MultiArray, 
                                                   self.callback_robot_pose_2d)
        
        self.lock_pedestrians = Lock()
        self.pedestrians = []
        self.leg_threshold = 1.5
        self.leg_width = 0.2
        self.single_human_width = 0.6

        self.lock_robot_pose = Lock()
        self.robot_pose = np.zeros(3)


    def callback_bounding_boxes(self, msg):
        size = len(msg.data)
        box_size = 6
        num_bounding_boxes = int(size / box_size)
        bounding_boxes = []
        for i in range(num_bounding_boxes):
            bounding_boxes.append(np.array([msg.data[box_size * i + 0],
                                            msg.data[box_size * i + 1],
                                            msg.data[box_size * i + 2],
                                            msg.data[box_size * i + 3],
                                            msg.data[box_size * i + 4],
                                            msg.data[box_size * i + 5]
                                            ], dtype=np.float32))
        self.pedestrians = []
        self.lock_pedestrians.acquire()
        possible_legs = []
        for i in range(num_bounding_boxes):
            size_x = fabs(bounding_boxes[i][3] - bounding_boxes[i][0])
            size_y = fabs(bounding_boxes[i][4] - bounding_boxes[i][1])
            if (size_x >= self.single_human_width) and (size_y >= self.single_human_width):
                position_x_1 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 - size_x / 4.0
                position_x_2 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 + size_x / 4.0
                position_y_1 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 - size_y / 4.0
                position_y_2 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 + size_y / 4.0
                self.pedestrians.append(np.array([position_x_1, position_y_1]))
                self.pedestrians.append(np.array([position_x_1, position_y_2]))
                self.pedestrians.append(np.array([position_x_2, position_y_1]))
                self.pedestrians.append(np.array([position_x_2, position_y_2]))
            elif (size_x >= self.single_human_width) and (size_y < self.single_human_width and size_y >= self.leg_width):
                position_x_1 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 - size_x / 4.0
                position_x_2 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 + size_x / 4.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                self.pedestrians.append(np.array([position_x_1, position_y]))
                self.pedestrians.append(np.array([position_x_2, position_y]))
            elif (size_x >= self.single_human_width) and (size_y < self.leg_width):
                position_x_1 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 - size_x / 4.0
                position_x_2 = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0 + size_x / 4.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                self.pedestrians.append(np.array([position_x_1, position_y]))
                self.pedestrians.append(np.array([position_x_2, position_y]))
            elif (size_x < self.single_human_width and size_x >= self.leg_width) and (size_y >= self.single_human_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y_1 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 - size_y / 4.0
                position_y_2 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 + size_y / 4.0
                self.pedestrians.append(np.array([position_x, position_y_1]))
                self.pedestrians.append(np.array([position_x, position_y_2]))
            elif (size_x < self.single_human_width and size_x >= self.leg_width) and (size_y < self.single_human_width and size_y >= self.leg_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                self.pedestrians.append(np.array([position_x, position_y]))
            elif (size_x < self.single_human_width and size_x >= self.leg_width) and (size_y < self.leg_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                self.pedestrians.append(np.array([position_x, position_y]))
            elif (size_x < self.leg_width) and (size_y >= self.single_human_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y_1 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 - size_y / 4.0
                position_y_2 = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0 + size_y / 4.0
                self.pedestrians.append(np.array([position_x, position_y_1]))
                self.pedestrians.append(np.array([position_x, position_y_2]))
            elif (size_x < self.leg_width) and (size_y < self.single_human_width and size_y >= self.leg_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                self.pedestrians.append(np.array([position_x, position_y]))
            elif (size_x < self.leg_width) and (size_y < self.leg_width):
                position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
                position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
                distance_robot = hypot(position_x - self.robot_pose[0], position_y - self.robot_pose[1])
                if distance_robot < self.leg_threshold:
                    possible_legs.append(np.array([position_x, position_y], dtype=np.float32))
                else:
                    self.pedestrians.append(np.array([position_x, position_y]))
        num_possible_legs = len(possible_legs)
        if num_possible_legs == 1:
            position_x = (bounding_boxes[i][3] + bounding_boxes[i][0]) / 2.0
            position_y = (bounding_boxes[i][4] + bounding_boxes[i][1]) / 2.0
            self.pedestrians.append(np.array([position_x, position_y]))
        elif num_possible_legs > 1:
            numpy_possible_legs = np.array(possible_legs)
            # Run DBSCAN to cluster
            db = DBSCAN(eps=self.leg_width*2, min_samples=1).fit(numpy_possible_legs)
            labels = db.labels_   

            unique_labels = np.unique(labels)
            for label in unique_labels:
                cluster_points = numpy_possible_legs[labels == label]
                center = np.mean(cluster_points, axis=0)
                self.pedestrians.append(center)
        self.lock_pedestrians.release()
        
    def callback_robot_pose_2d(self, msg):
        self.lock_robot_pose.acquire()
        self.robot_pose[0] = msg.data[0]
        self.robot_pose[1] = msg.data[1]
        self.robot_pose[2] = msg.data[2]
        self.lock_robot_pose.release()


if __name__ == "__main__":
    rospy.init_node("subscribe_bounding_boxes", anonymous=True)
    node = SubBoundingBox()
    rospy.spin()
