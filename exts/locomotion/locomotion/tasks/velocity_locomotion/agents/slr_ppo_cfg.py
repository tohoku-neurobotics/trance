from isaaclab.utils import configclass

@configclass
class AlgorithmCfg:
    entropy_coef: float = 0.01
    learning_rate: float = 1e-3
    max_grad_norm: float = 1
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    schedule: str = 'adaptive'
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01

@configclass
class PolicyCfg:
    init_noise_std: float = 1.0
    continue_from_last_std: bool = True
    scan_encoder_dims: list = []
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    priv_encoder_dims: list = []
    activation: str = 'elu'
    rnn_type: str = 'lstm'
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    tanh_encoder_output: bool = False

@configclass
class Go2RoughPPOCfg:
    device: str = 'cuda:0'
    seed: int = 1
    run_name: str = 'slr'
    experiment_name: str = 'rough_go2'
    policy_class_name: str = 'ActorCriticSLR'
    algorithm_class_name: str = 'PPO'
    max_iterations: int = 6000
    num_steps_per_env: int = 24
    resume: str = None
    load_run: str = ''
    load_checkpoint: str = ''
    save_interval: int = 100
    algorithm: AlgorithmCfg = AlgorithmCfg()
    policy: PolicyCfg = PolicyCfg()
    clip_actions: float | None = None