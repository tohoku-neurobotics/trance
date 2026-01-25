from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np

experiment_name = "rough_go2"
log_root_path = os.path.join("logs", "dream_waq_v2", experiment_name)
log_root_path = os.path.abspath(log_root_path)
load_run = "2025-09-27_21-30-17_dream_waq_cnn_decoder_more_random"  
load_tensorboard_data = "events.out.tfevents.1758976230.wei.388334.0"

event_path = os.path.join(log_root_path, load_run, load_tensorboard_data)
# event_path = "/home/wei/Downloads/IsaacLab/logs/rsl_rl/unitree_go2_rough/2025-08-29_10-43-30/events.out.tfevents.1756431818.wei.610215.0"

# Load the event file
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# List available keys (tags)
print("Available tags:", ea.Tags())

# Extract scalar values for a specific tag
tag = "Episode/Curriculum/terrain_levels"   # <-- change this to your key
# tag = "Curriculum/terrain_levels"   # <-- change this to your key
scalars = ea.Scalars(tag)

data_save = []

# scalars is a list of Event objects: (wall_time, step, value)
for s in scalars:
    # print(f"step={s.step}, value={s.value}")
    if s.step <= 2999:
        data_save.append(np.array([s.step, s.value], dtype=np.float32))
np.savetxt('rough_go2_more_random.txt', np.array(data_save))