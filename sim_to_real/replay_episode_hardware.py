import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import time
from math import cos, sin, hypot, fabs

from lidar_reconstruction.motion_plan_lib_opt import *




plt.rcParams['font.family'] = 'Times New Roman'

use_laser = True

n_laser = 1800

log_file = 'log_20251217_103517'

specific_file_name = './evaluation_episodes/' + log_file + '.npz'


log_env = np.load(specific_file_name)

robot = log_env['robot']
steps = robot.shape[0]
humans = log_env['humans']

human_num = humans.shape[1]
goal = log_env['goal']

radius_robot = 0.3


#####real time plot for simulation case########
# radius = 0.3
# plt.ion()
# plt.show()
# fig, ax = plt.subplots(figsize=(10, 10))
# for i in range(steps):
#     if fabs(robot[i][0]) > 50.0:
#         break
#     ax.set_xlim(-1.0, 8.0)
#     ax.set_ylim(-4.5, 4.5)
#     if use_laser:
#         scan_intersection = []
#         robot_pose_new = robot[i]
#         human_circles = humans[i]
#         scans_raw, scan_lines = cal_laser(human_circles, robot_pose_new)

#         for laser_i in range(n_laser):
#             scan_intersection.append([(scan_lines[laser_i][0], scan_lines[laser_i][1]), 
#                                       (scan_lines[laser_i][2], scan_lines[laser_i][3])])
#     for human_i in range(human_num):
#         if fabs(humans[i][human_i, 0]) > 50.0:
#             break
#         human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
#         ax.add_artist(human_circle)
#     ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
#     plt.text(7.0, 3.5, str(round(i * 0.2, 2)), fontsize=20)

#     x, y, theta = robot[i][0], robot[i][1], robot[i][2]
#     print('robot position: ', x, y)
#     dx = cos(theta)
#     dy = sin(theta)
#     ax.arrow(x, y, dx, dy,
#         width=0.01,
#         length_includes_head=True, 
#         head_width=0.15,
#         head_length=1,
#         fc='r',
#         ec='r')
    
#     if use_laser:
#         ii = 0
#         lines = []
#         while ii < n_laser:
#             lines.append(scan_intersection[ii])
#             ii = ii + 36
#         lc = mc.LineCollection(lines)
#         ax.add_collection(lc)
#     plt.draw()
#     plt.pause(0.001)
#     plt.cla()
#     time.sleep(0.2)
####real time plot for simulation case########



# ##### video ########
radius = 0.3
metadata = dict(title='Trance', artist='Matplotlib',comment='Trance test')
writer = FFMpegWriter(fps=5, metadata=metadata)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
plt.tick_params(labelsize=24)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]


ax.set_xlim(-1.0, 8.0)
ax.set_ylim(-4.5, 4.5)

video_name = './evaluation_episodes/' + log_file + '.mp4'
 
with writer.saving(fig, video_name, 250):
    for i in range(steps):
        if fabs(robot[i][0]) > 50.0:
            break
        ax.clear()
        ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
        ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
        plt.tick_params(labelsize=24)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        
        ax.set_xlim(-1.0, 8.0)
        ax.set_ylim(-4.5, 4.5)


        if use_laser:
            scan_intersection = []
            robot_pose_new = robot[i]
            human_circles = humans[i]
            scans_raw, scan_lines = cal_laser(human_circles, robot_pose_new)

            for laser_i in range(n_laser):
                scan_intersection.append([(scan_lines[laser_i][0], scan_lines[laser_i][1]), 
                                        (scan_lines[laser_i][2], scan_lines[laser_i][3])])
            
        for human_i in range(human_num):
            if fabs(humans[i][human_i, 0]) > 50.0:
                break
            human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
            ax.add_artist(human_circle)    
        ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
        ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

        x, y, theta = robot[i][0], robot[i][1], robot[i][2]
        dx = cos(theta)
        dy = sin(theta)
        ax.arrow(x, y, dx, dy,
            width=0.01,
            length_includes_head=True, 
            head_width=0.15,
            head_length=1,
            fc='r',
            ec='r')
    

        if use_laser:
            ii = 0
            lines = []
            while ii < n_laser:
                lines.append(scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            ax.add_collection(lc)

        ax.text(7.0, 3.5, str(round(i * 0.2, 2)), fontsize=20)
        writer.grab_frame()

# if ffmeg error occurs
# do not use anaconda env