import torch
from rsl_rl.networks import MLP, EmpiricalNormalization

# other useful RSL RL imports
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic


ckpt_path = '/home/mpan31415/rl_repos/IsaacLabTest/ckpt_testing/xela_data_collector/norm_obs_z_axis/'

ckpt_name = 'model_13500.pt'

# define dimensions
OBS_SPACE_DIM = 96
ACTION_SPACE_DIM = 16
ACTOR_HIDDEN_DIMS = [512, 256, 128]


# 1. load checkpoint
loaded_dict = torch.load(ckpt_path + ckpt_name, weights_only=False)
loaded_model_state_dict = loaded_dict['model_state_dict']

# 2. create actor and load weights
actor = MLP(input_dim=OBS_SPACE_DIM, output_dim=ACTION_SPACE_DIM, hidden_dims=ACTOR_HIDDEN_DIMS, activation='elu')
loaded_actor_dict = {k.replace("actor.", "", 1): v 
                    for k, v in loaded_model_state_dict.items() 
                    if k.startswith("actor.")}
actor.load_state_dict(loaded_actor_dict)

# 3. create actor obs normalizer and load weights
actor_obs_normalizer = EmpiricalNormalization(shape=OBS_SPACE_DIM)
loaded_actor_obs_normalizer_dict = {k.replace("actor_obs_normalizer.", "", 1): v 
                                    for k, v in loaded_model_state_dict.items() 
                                    if k.startswith("actor_obs_normalizer.")}
actor_obs_normalizer.load_state_dict(loaded_actor_obs_normalizer_dict)


# 4. test the loaded actor and normalizer
actor.eval()
actor_obs_normalizer.eval()

test_input = torch.zeros((1, 96))

normalized_input = actor_obs_normalizer(test_input)
output = actor(normalized_input)
print("Output from loaded actor:", output)
