#!/usr/bin/env python3
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

GRID_SIZE      = 10.0
STEP_SIZE      = 0.4
LIFT_THRESHOLD = 0.5
MAX_STEPS      = 50           # short episodes = more episodes = faster learning
SPAWN_RADIUS   = 0.8          # spawn close so random policy discovers lift

_R = 0.5
BOX_POS = np.array([5.0, 5.0])
TARGET_POSITIONS = np.array([
    BOX_POS + [_R * math.cos(math.radians(90)),   _R * math.sin(math.radians(90))],
    BOX_POS + [_R * math.cos(math.radians(210)),  _R * math.sin(math.radians(210))],
    BOX_POS + [_R * math.cos(math.radians(330)),  _R * math.sin(math.radians(330))],
])

N_ROBOTS = 3
ACTIONS  = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]], dtype=float) * STEP_SIZE


class BoxLiftEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # obs: for each robot → [rel_x, rel_y, distance]  (3 values × 3 robots = 9)
        self.observation_space = spaces.Box(
            low=-GRID_SIZE * 2,
            high=GRID_SIZE * 2,
            shape=(N_ROBOTS * 3,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        self._robot_pos = np.zeros((N_ROBOTS, 2), dtype=np.float32)
        self._lifted    = False
        self._steps     = 0

    def _get_obs(self):
        obs = []
        for i in range(N_ROBOTS):
            rel  = (TARGET_POSITIONS[i] - self._robot_pos[i]).astype(np.float32)
            dist = float(np.linalg.norm(rel))
            obs += [rel[0], rel[1], dist]
        return np.array(obs, dtype=np.float32)

    def _dist(self, i):
        return float(np.linalg.norm(self._robot_pos[i] - TARGET_POSITIONS[i]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        for i in range(N_ROBOTS):
            angle = rng.uniform(0, 2 * math.pi)
            r     = rng.uniform(0.3, SPAWN_RADIUS)
            pos   = TARGET_POSITIONS[i] + r * np.array([math.cos(angle), math.sin(angle)])
            self._robot_pos[i] = np.clip(pos, 0.0, GRID_SIZE).astype(np.float32)
        self._lifted = False
        self._steps  = 0
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1

        for i in range(N_ROBOTS):
            new_pos = self._robot_pos[i] + ACTIONS[action[i]]
            self._robot_pos[i] = np.clip(new_pos, 0.0, GRID_SIZE).astype(np.float32)

        # Per-robot reward: +1 if in position, -dist*0.1 otherwise
        # This gives clear gradient AND makes being-in-position the desired behavior
        reward = 0.0
        n_in   = 0
        for i in range(N_ROBOTS):
            d = self._dist(i)
            if d <= LIFT_THRESHOLD:
                reward += 1.0
                n_in   += 1
            else:
                reward -= d * 0.1

        # Terminal bonus — large enough to matter
        terminated = False
        if n_in == N_ROBOTS:
            self._lifted  = True
            reward        += 50.0
            terminated    = True

        truncated = (self._steps >= MAX_STEPS)
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Step {self._steps} ---")
        for i in range(N_ROBOTS):
            print(f"  Robot {i+1}: dist={self._dist(i):.3f}  "
                  f"in_pos={self._dist(i) <= LIFT_THRESHOLD}")
        print(f"  Lifted: {self._lifted}")