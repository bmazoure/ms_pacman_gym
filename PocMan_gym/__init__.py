from gym.envs.registration import register

from .pocman_env import PocMan
import os 
is_windows = os.name == 'nt'
sep = '/' if not is_windows else '\\'

__all__ = ['PocMan']

register(
    id='PocManSparseScalar-v0',
    entry_point='.pocman_env.envs:PocMan',
    kwargs={'observation_type':'sparse_scalar'}
)

register(
    id='PocManSparseVector-v0',
    entry_point='pocman_env.envs:PocMan',
    kwargs={'observation_type':'sparse_vector'}
)