from gym.envs.registration import register

import os 
is_windows = os.name == 'nt'
sep = '/' if not is_windows else '\\'

__all__ = ['PocMan']

register(
    id='PocManSparseScalar-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'sparse_scalar'}
)

register(
    id='PocManSparseVector-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'sparse_vector'}
)

register(
    id='PocManFullScalar-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'full_scalar'}
)

register(
    id='PocManFullector-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'full_vector'}
)

register(
    id='PocManFullASCII-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'full_ascii'}
)

register(
    id='PocManFullRGB-v0',
    entry_point='pocman_gym.envs:PocMan',
    kwargs={'observation_type':'full_rgb'}
)