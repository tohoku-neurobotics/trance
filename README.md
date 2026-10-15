# Project info
- [website](https://tohoku-neurobotics.github.io/trance.github.io/)
- This work has been accepted by IEEE Transactions on Automation Science and Engineering

# Training environments
- ubuntu 24
- NVIDIA RTX 5080
- Isaaclab 2.2, IsaacSim 5.0


## Locomotion
- install RLopt, `git clone https://github.com/fei-yang-wu/RLOpt.git`, `pip install -e .`
- install locomotion package
```bash
pip install stable-baselines3[extra]
python -m pip install -e exts/locomotion
pip install -e .
```
- run locomotion
`python scripts/dream_waq/train_v2.py --task Dream-WaQ-Rough-Unitree-Go2-v2 --num_envs 4096 --headless`
- run l2t baseline
`python scripts/rlopt/train.py --task Isaac-Velocity-Rough-Go2-L2T-v0 --num_envs 4096 --headless`
- evaluate locomotion with visualization
`python scripts/dream_waq/play_v2.py --task Dream-WaQ-Rough-Unitree-Go2-v2 --num_envs 128`

## Navigation
- in DRL_dynamic_navigation/C_library, `python setup.py build_ext --inplace`
- intall [ORCA](https://github.com/sybrenstuvel/Python-RVO2)
- please set robot_model as differential, so you can train your crowd navigation fast without using quadruped robot, e.g. `python soc_qua_dream_waq_main.py --robot_model differential`
- after training with differential robot model, use the best checkpoint as warm-start for your quadrupedal navigation policy
- change log_root_path in load_quadruped_policy.py to load your locomotion policy
- change to old_agent.load() in soc_qua_dream_waq_main.py load your pretrained crowd navigation policy
- retrain the unified policy, `python soc_qua_dream_waq_main.py --robot_model quadruped --headless`, this takes time!
- evaluate your policy, just change here `parser.add_argument("--load_test_model", type=str, default="step_1000000")`



