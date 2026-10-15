import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import glob
import time
from math import cos, sin, hypot, fabs

import psutil
import os


n_laser = 1800


complex_env = True
use_laser = True

# specific_file_name = './logs/sac_differential_transform/seed_1/final_test_episodes_600/eval_0_23.npz'
# specific_file_name = './logs/tmpc_differential/seed_1/final_test_episodes/eval_23.npz'
# specific_file_name = './logs/dwa_differential/seed_1/final_test_episodes/eval_0_23.npz'
# specific_file_name = './logs/orca_differential/seed_1/final_test_episodes/eval_0_23.npz'
# specific_file_name = './logs/td3_lstm_differential/seed_1/final_test_episodes_40/eval_0_23.npz'
# specific_file_name = './logs/drl_vo_differential_random_radius/seed_1/final_test_episodes_598/eval_0_23.npz'

# specific_file_name = './logs/rgl_differential_random_radius/seed_1/final_test_episodes/eval_1340229_23_fail.npz'
# specific_file_name = './logs/avocado_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'
# specific_file_name = './logs/neupan_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'
# specific_file_name = './logs/nmpc_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'


# 12 (fall), 87 (fall), 103 (fall), 458, 474
# 48 (stuck), 82 (stuck), 211, 226, 295, 324, 368, 398, 480
# 98 (collision), 102 (collision), 180(collision), 222, 261, 329, 359, 371, 426, 453, 
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_952_stable/eval_0_453_fail.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000/eval_0_30_fail.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64_obs_bug/seed_1/final_test_episodes_step_1140000/eval_0_84_fail.npz'

# specific_file_name = "./logs/sac_quadruped_transform_dream_waq_cnn_penalty_rotation_64/seed_1/final_test_episodes_step_1300000_962/eval_0_2.npz"

# 41, 198, 328, 353, 387, 472, collision and pedestrian fault
# 113, 116, 334, collision and robot fault
# 127, 286, 469, collision and both fault
# 0, 2, 8, 11, 13, 43
# specific_file_name = \
#     "./logs/sac_quadruped_transform_dream_waq_cnn_penalty_rotation_64/seed_1/final_test_episodes_step_1460000/eval_0_43.npz"
specific_file_name = \
    "./logs/sac_differential_transform_backward/seed_1/final_test_episodes_step_3260000_success_100-980/eval_0_6.npz"
log_env = np.load(specific_file_name)

robot = log_env['robot']
steps = robot.shape[0]
humans = log_env['humans']
if use_laser:
    laser = log_env['laser']
    laser_beam = laser.shape[1]
human_num = humans.shape[1]
goal = log_env['goal']

static_obstacles = []
radius_robot = 0.3


#####real time plot for simulation case########
# radius = 0.3
# plt.ion()
# plt.show()
# fig, ax = plt.subplots(figsize=(10, 10))
# for i in range(steps):
#     if fabs(robot[i][0]) > 50.0:
#         break
#     ax.set_xlim(-6.0, 6.0)
#     ax.set_ylim(-6.0, 6.0)
#     if use_laser:
#         scan_intersection = []
#         for laser_i in range(laser_beam):
#             scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
#         ax.add_artist(human_circle)
#     ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
#     plt.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)
#     print('action: ', robot[i][2], robot[i][3])
#     if complex_env:
#         static_obstacles = log_env['static_obstacles']
#         static_obstacle_num = static_obstacles.shape[1]
#         for static_i in range(static_obstacle_num):
#             ax.add_artist(plt.Circle((static_obstacles[i][static_i, 0], static_obstacles[i][static_i, 1]), 
#                                       static_obstacles[i][static_i, 2],
#                                       fill=True, color='c'))

#     x, y, theta = robot[i][0], robot[i][1], robot[i][4]
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

###### save pdf for simulation case #########
# radius = 0.3
# for i in range(steps):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_xlim(-5.0, 5.0)
#     ax.set_ylim(-5.0, 5.0)
#     scan_intersection = []
#     for laser_i in range(laser_beam):
#         scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
#         ax.add_artist(human_circle)

#     ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

#     if complex_env:
#         static_obstacles = log_env['static_obstacles']
#         static_obstacle_num = static_obstacles.shape[1]
#         for static_i in range(static_obstacle_num):
#             ax.add_artist(plt.Circle((static_obstacles[i][static_i, 0], static_obstacles[i][static_i, 1]), 
#                                       static_obstacles[i][static_i, 2],
#                                       fill=True, color='c'))
            
#     # plt.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)

#     ii = 0
#     lines = []
#     while ii < n_laser:
#         lines.append(scan_intersection[ii])
#         ii = ii + 36
#     lc = mc.LineCollection(lines)
#     ax.add_collection(lc)
#     plt.savefig("SimImagePDF_" + str(i) + ".pdf", format="pdf", bbox_inches="tight")
#     plt.show()
# ####### save pdf for simulation case #########

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



ax.set_xlim(-6.0, 6.0)
ax.set_ylim(-6.0, 6.0)
 
with writer.saving(fig, "backward_motion.mp4", 250):
    for i in range(steps):
        if fabs(robot[i][0]) > 50.0:
            break
        ax.clear()
        ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
        ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
        plt.tick_params(labelsize=24)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        
        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-6.0, 6.0)

        if use_laser:
            scan_intersection = []
            for laser_i in range(laser_beam):
                scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])

        for human_i in range(human_num):
            human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
            ax.add_artist(human_circle)
        ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
        ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

        x, y, theta = robot[i][0], robot[i][1], robot[i][4]
        dx = cos(theta)
        dy = sin(theta)
        ax.arrow(x, y, dx, dy,
            width=0.01,
            length_includes_head=True, 
            head_width=0.15,
            head_length=1,
            fc='r',
            ec='r')
    
        if complex_env:
            static_obstacles = log_env['static_obstacles']
            static_obstacle_num = static_obstacles.shape[1]
            for static_i in range(static_obstacle_num):
                ax.add_artist(plt.Circle((static_obstacles[i][static_i, 0], static_obstacles[i][static_i, 1]), 
                                        static_obstacles[i][static_i, 2],
                                        fill=True, color='c'))
        if use_laser:
            ii = 0
            lines = []
            while ii < n_laser:
                lines.append(scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            ax.add_collection(lc)

        ax.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)
        writer.grab_frame()

# if ffmeg error occurs
# do not use anaconda env