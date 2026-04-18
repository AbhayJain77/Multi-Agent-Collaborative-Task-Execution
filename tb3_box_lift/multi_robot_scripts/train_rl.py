#!/usr/bin/env python3
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

_R = 0.6
N_ROBOTS       = 3
STEP_SIZE      = 0.2
LIFT_THRESHOLD = 0.5
MAX_STEPS      = 300
ACTIONS        = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]], dtype=float) * STEP_SIZE


class BoxLiftEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(N_ROBOTS * 2,), dtype=np.float32
        )
        self.action_space      = spaces.MultiDiscrete([5, 5, 5])
        self._robot_pos        = np.zeros((N_ROBOTS, 2), dtype=np.float32)
        self._targets          = np.zeros((N_ROBOTS, 2), dtype=np.float32)
        self._box_pos          = np.zeros(2, dtype=np.float32)
        self._lifted           = False
        self._steps            = 0

    def _get_obs(self):
        obs = []
        for i in range(N_ROBOTS):
            rel = (self._targets[i] - self._robot_pos[i]).astype(np.float32)
            obs += [rel[0], rel[1]]
        return np.array(obs, dtype=np.float32)

    def _dist(self, i):
        return float(np.linalg.norm(self._robot_pos[i] - self._targets[i]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Random box position every episode
        self._box_pos = rng.uniform(-5.0, 5.0, size=2).astype(np.float32)

        # Targets around random box
        self._targets = np.array([
            self._box_pos + np.array([ 0.0,   _R   ]),
            self._box_pos + np.array([-_R,   -_R/2 ]),
            self._box_pos + np.array([ _R,   -_R/2 ]),
        ], dtype=np.float32)

        # Robots spawn randomly around their targets
        for i in range(N_ROBOTS):
            angle = rng.uniform(0, 2 * math.pi)
            r     = rng.uniform(2.0, 8.0)
            pos   = self._targets[i] + r * np.array([math.cos(angle),
                                                      math.sin(angle)])
            self._robot_pos[i] = np.clip(pos, -15.0, 15.0).astype(np.float32)

        self._lifted = False
        self._steps  = 0
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        for i in range(N_ROBOTS):
            new_pos = self._robot_pos[i] + ACTIONS[action[i]]
            self._robot_pos[i] = np.clip(new_pos, -15.0, 15.0).astype(np.float32)

        total_dist = sum(self._dist(i) for i in range(N_ROBOTS))
        reward     = -total_dist * 0.1

        terminated = False
        if all(self._dist(i) <= LIFT_THRESHOLD for i in range(N_ROBOTS)):
            self._lifted = True
            reward      += 500.0
            terminated   = True

        truncated = (self._steps >= MAX_STEPS)
        return self._get_obs(), reward, terminated, truncated, {}


def sanity_check():
    print("Running sanity check...")
    env = BoxLiftEnv()
    env.reset(seed=42)
    env._robot_pos = env._targets.copy()
    _, reward, terminated, _, _ = env.step([0, 0, 0])
    assert terminated, "FAIL"
    print(f"  PASSED — reward={reward:.1f}")


def main():
    sanity_check()
    os.makedirs("./logs",        exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    env      = BoxLiftEnv()
    eval_env = BoxLiftEnv()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./box_lift_best",
        log_path="./logs",
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 1024,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        ent_coef        = 0.05,
        verbose         = 1,
        tensorboard_log = "./logs",
    )

    print("\nTraining with randomized box positions...\n")
    model.learn(total_timesteps=500_000, callback=eval_cb, progress_bar=True)
    model.save("box_lift_final")
    print("\nSaved: box_lift_final.zip")

    print("\n--- Evaluation ---")
    successes = 0
    for ep in range(10):
        obs, _  = eval_env.reset()
        done    = False
        total_r = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(act)
            total_r += r
            done = term or trunc
        if eval_env._lifted:
            successes += 1
        print(f"  Ep {ep+1}: reward={total_r:.1f}  lifted={eval_env._lifted}")
    print(f"\nSuccess rate: {successes}/10")


if __name__ == "__main__":
    main()