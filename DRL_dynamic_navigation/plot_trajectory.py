import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import glob
import time
from math import cos, sin, hypot, fabs



# specific_file_name = './logs/tmpc_differential/seed_1/final_test_episodes/eval_23.npz'
# specific_file_name = './logs/dwa_differential/seed_1/final_test_episodes/eval_0_23.npz'
# specific_file_name = './logs/orca_differential/seed_1/final_test_episodes/eval_0_23.npz'
# specific_file_name = './logs/td3_lstm_differential/seed_1/final_test_episodes_40/eval_0_23.npz'
# specific_file_name = './logs/drl_vo_differential_random_radius/seed_1/final_test_episodes_598/eval_0_23.npz'

# specific_file_name = './logs/rgl_differential_random_radius/seed_1/final_test_episodes/eval_1340229_23_fail.npz'
# specific_file_name = './logs/avocado_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'
# specific_file_name = './logs/neupan_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'
# specific_file_name = './logs/nmpc_differential/seed_1/final_test_episodes/eval_0_23_fail.npz'
# specific_file_name = './logs/sac_differential_transform/seed_1_frame_6/final_test_episodes_step_3820000_success_98-972/eval_0_23.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_1/eval_0_261_fail.npz'

# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_3/eval_0_23_fail.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_952_stable/eval_0_329_fail.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000/eval_0_30_fail.npz'
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_1/eval_0_261_fail.npz'  # fall
# specific_file_name = './logs/sac_quadruped_transform_dream_waq_cnn_64/seed_1/final_test_episodes_step_1060000_3/eval_0_23_fail.npz'  # timeout

specific_file_name = \
    "./logs/sac_quadruped_transform_dream_waq_cnn_penalty_rotation_64/seed_1/final_test_episodes_step_1460000/eval_0_32_fail.npz"

log_env = np.load(specific_file_name)

robot = log_env['robot']
steps = robot.shape[0] 
# steps = 80
humans = log_env['humans']

human_num = humans.shape[1]
goal = log_env['goal']

static_obstacles = []
radius_robot = 0.3

# Set global font and label size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

human_color = ['b', 'm', 'y', 'k']

# plot multiple sub-figures
subfigure_num = 3
idx_delta = 3
fig2, axes2 = plt.subplots(1, subfigure_num, figsize=(4 * subfigure_num, 4))

# Set global limits for all subplots
for ax2 in axes2:
    ax2.set_xlim(-5.0, 5.0)
    ax2.set_ylim(-5.0, 5.0)   # example range, adjust as needed

total_step = 0
for i in range(steps):
    if fabs(robot[i][0]) > 50.0:
        break
    total_step = total_step + 1

interval = int(total_step / subfigure_num)

plot_idx = 0
while plot_idx < total_step:
    subfigure_idx = int(plot_idx / interval)
    if subfigure_idx >= subfigure_num:
        subfigure_idx = subfigure_num - 1
    alpha = (plot_idx + 1) / total_step

    for human_i in range(human_num):
        human_circle = plt.Circle(humans[plot_idx][human_i, 0:2], humans[plot_idx][human_i, 2], 
                                  fill=True, alpha=alpha, color=human_color[human_i])
        axes2[subfigure_idx].add_artist(human_circle)

    axes2[subfigure_idx].add_artist(plt.Circle((robot[plot_idx][0], robot[plot_idx][1]), 
                                               radius_robot, fill=True, alpha=alpha, color='r'))
    axes2[subfigure_idx].add_artist(plt.Circle(goal[plot_idx], radius_robot, fill=True, color='g'))

    static_obstacles = log_env['static_obstacles']
    static_obstacle_num = static_obstacles.shape[1]
    for static_i in range(static_obstacle_num):
        axes2[subfigure_idx].add_artist(plt.Circle((static_obstacles[plot_idx][static_i, 0], static_obstacles[plot_idx][static_i, 1]), 
                                            static_obstacles[plot_idx][static_i, 2],
                                            fill=True, color='c'))

    plot_idx = plot_idx + idx_delta

for human_i in range(human_num):
    human_circle = plt.Circle(humans[total_step - 1][human_i, 0:2], humans[total_step - 1][human_i, 2], 
                                fill=True, alpha=alpha, color=human_color[human_i])
    axes2[subfigure_idx].add_artist(human_circle)

axes2[subfigure_idx].add_artist(plt.Circle((robot[total_step - 1][0], robot[total_step - 1][1]), 
                                            radius_robot, fill=True, alpha=alpha, color='r'))
axes2[subfigure_idx].add_artist(plt.Circle(goal[total_step - 1], radius_robot, fill=True, color='g'))

static_obstacles = log_env['static_obstacles']
static_obstacle_num = static_obstacles.shape[1]
for static_i in range(static_obstacle_num):
    axes2[subfigure_idx].add_artist(plt.Circle((static_obstacles[total_step - 1][static_i, 0], static_obstacles[total_step - 1][static_i, 1]), 
                                        static_obstacles[total_step - 1][static_i, 2],
                                        fill=True, color='c'))
plt.tight_layout()
plt.show()
# plot multiple sub-figures


# plot single figure
# idx_delta = 4
# fig2, ax2 = plt.subplots(figsize=(5, 5))

# ax2.set_xlim(-5.0, 5.0)
# ax2.set_ylim(-5.0, 5.0)   

# total_step = 0
# for i in range(steps):
#     if fabs(robot[i][0]) > 50.0:
#         break
#     total_step = total_step + 1

# plot_idx = 0
# while plot_idx < total_step:
#     alpha = (plot_idx + 1) / total_step
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[plot_idx][human_i, 0:2], humans[plot_idx][human_i, 2], 
#                                   fill=True, alpha=alpha, color=human_color[human_i])
#         ax2.add_artist(human_circle)

#     ax2.add_artist(plt.Circle((robot[plot_idx][0], robot[plot_idx][1]), 
#                                                radius_robot, fill=True, alpha=alpha, color='r'))
#     ax2.add_artist(plt.Circle(goal[plot_idx], radius_robot, fill=True, color='g'))

#     static_obstacles = log_env['static_obstacles']
#     static_obstacle_num = static_obstacles.shape[1]
#     for static_i in range(static_obstacle_num):
#         ax2.add_artist(plt.Circle((static_obstacles[plot_idx][static_i, 0], static_obstacles[plot_idx][static_i, 1]), 
#                                             static_obstacles[plot_idx][static_i, 2],
#                                             fill=True, color='c'))

#     plot_idx = plot_idx + idx_delta

# for human_i in range(human_num):
#     human_circle = plt.Circle(humans[total_step - 1][human_i, 0:2], humans[total_step - 1][human_i, 2], 
#                                 fill=True, alpha=alpha, color=human_color[human_i])
#     ax2.add_artist(human_circle)

# ax2.add_artist(plt.Circle((robot[total_step - 1][0], robot[total_step - 1][1]), 
#                                             radius_robot, fill=True, alpha=alpha, color='r'))
# ax2.add_artist(plt.Circle(goal[total_step - 1], radius_robot, fill=True, color='g'))

# static_obstacles = log_env['static_obstacles']
# static_obstacle_num = static_obstacles.shape[1]
# for static_i in range(static_obstacle_num):
#     ax2.add_artist(plt.Circle((static_obstacles[total_step - 1][static_i, 0], static_obstacles[total_step - 1][static_i, 1]), 
#                                         static_obstacles[total_step - 1][static_i, 2],
#                                         fill=True, color='c'))
# plt.tight_layout()
# plt.show()
# plot single figure

# save video
# radius = 0.3
# metadata = dict(title='Trance', artist='Matplotlib',comment='Trance test')
# writer = FFMpegWriter(fps=5, metadata=metadata)

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
# ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
# plt.tick_params(labelsize=24)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname("Times New Roman") for label in labels]


# ax.set_xlim(-5.0, 5.0)
# ax.set_ylim(-5.0, 5.0)
 
# with writer.saving(fig, "robot_fault.mp4", 250):
#     for i in range(steps):
#         if fabs(robot[i][0]) > 50.0:
#             break
#         ax.clear()
#         ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
#         ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
#         plt.tick_params(labelsize=24)
#         labels = ax.get_xticklabels() + ax.get_yticklabels()
#         [label.set_fontname("Times New Roman") for label in labels]
        
#         ax.set_xlim(-5.0, 5.0)
#         ax.set_ylim(-5.0, 5.0)

#         for human_i in range(human_num):
#             human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color='b')
#             ax.add_artist(human_circle)
#         ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
#         ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

#         x, y, theta = robot[i][0], robot[i][1], robot[i][4]
#         dx = cos(theta)
#         dy = sin(theta)
#         ax.arrow(x, y, dx, dy,
#             width=0.01,
#             length_includes_head=True, 
#             head_width=0.15,
#             head_length=1,
#             fc='r',
#             ec='r')
    
  
#         static_obstacles = log_env['static_obstacles']
#         static_obstacle_num = static_obstacles.shape[1]
#         for static_i in range(static_obstacle_num):
#             ax.add_artist(plt.Circle((static_obstacles[i][static_i, 0], static_obstacles[i][static_i, 1]), 
#                                     static_obstacles[i][static_i, 2],
#                                     fill=True, color='c'))

#         ax.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)
#         writer.grab_frame()
# save video
