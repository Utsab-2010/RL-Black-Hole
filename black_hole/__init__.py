from gymnasium.envs.registration import register
from .game import BlackHoleGame
from .env import BlackHoleEnv

register(
    id="BlackHole-v0",
    entry_point="black_hole.env:BlackHoleEnv",
)
