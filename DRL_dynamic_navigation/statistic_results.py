import numpy as np
import glob
from math import fabs
import statistics as stats

# specific_folder = './logs/avocado_differential/seed_1/final_test_episodes/*.npz'
# specific_folder = './logs/sac_quadruped_transform_dream_waq_terrain_enabled/seed_1/final_test_episodes_init/*.npz'
# specific_folder = './logs/sac_differential_transform_domain_randomization/seed_1/final_test_episodes_step_4240000_success_99_962/*.npz'

# specific_folder = './logs/sac_differential_direct/seed_1/final_test_episodes_104/*.npz'

# specific_folder = './logs/sac_differential_transform/seed_1_frame_6/final_test_episodes_step_3680000_success_99/*.npz'
# specific_folder = './logs/sac_differential_transform_backward/seed_1/final_test_episodes_step_3260000_success_100-980/*.npz'
# specific_folder = './logs/sac_quadruped_transform_dream_waq_cnn_penalty_rotation_64/seed_1_clip_in_place_rotation/final_test_episodes_step_1380000-968/*.npz'
# specific_folder = './logs/height/*.npz'
specific_folder = './logs/rgl_differential_random_radius/seed_4/final_test_episodes-786/*.npz'

file_list = glob.glob(specific_folder)

count_success = 0
count_timeout = 0
nav_time = []
for file_name in file_list:
    nav_count = 0
    if ('fail' not in file_name) and ('fall' not in file_name):
        count_success = count_success + 1
        nav_data = np.load(file_name)
        for i in range(nav_data['robot'].shape[0]):
            if fabs(nav_data['robot'][i][0]) > 50.0:
                break
            nav_count += 1
        nav_time.append(nav_count * 0.2)
    else:
        fail_time = 250
        nav_data = np.load(file_name)
        for i in range(nav_data['robot'].shape[0]):
            if fabs(nav_data['robot'][i][0]) > 50.0:
                fail_time = i
                break
        if fail_time >= 250:
            count_timeout = count_timeout + 1
            
len_file_list = len(file_list)
if len_file_list > 0:
    print('success_rate: ', count_success / len(file_list))
print('collision_rate: ', (len_file_list - count_success - count_timeout) / len(file_list))
print('timeout_rate: ', count_timeout / len(file_list))

len_nav_time = len(nav_time)
if len_nav_time > 0:
    print('navigation time mean: ', stats.mean(nav_time))
    print('navigation time std: ', stats.stdev(nav_time))