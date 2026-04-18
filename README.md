# tb3_box_lift — Multi-Agent RL Collaborative Box Lifting

Three TurtleBot3 robots learn to collaboratively position themselves around a box using **Reinforcement Learning (PPO)**. Each robot has limited upward force — the box lifts only when all 3 reach correct positions simultaneously. The RL model is trained in a custom Gymnasium environment and deployed on real Gazebo simulation via ROS2.

---

## Project Structure
```
tb3_box_lift/
├── multi_robot_scripts/
│   ├── train_rl.py
│   └── box_lift_rl_controller.py
├── launch/
│   └── tb3_world.launch.py
├── config/
│   └── robots.yaml
├── worlds/
│   └── tb3_world.world
└── restart_sim.sh
```

---

## RL Design

| Component | Detail |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Observation | Relative vector `[rel_x, rel_y]` per robot to its target = 6 values |
| Action | `MultiDiscrete [5,5,5]` — stay/±x/±y per robot |
| Reward | `-distance × 0.1` per step + `+500` terminal bonus when all 3 lift |
| Target layout | Equilateral triangle (r=0.6m) around box |
| Training steps | 500,000 timesteps |
| Success rate | 10/10 in evaluation |

---

## Proof That Solution Is NOT Hardcoded

```python
def gz_get_pose(model_name):
    result = subprocess.run(["gz", "model", "-m", model_name, "-p"], ...)
```

- Box position is read dynamically
- Robot spawn positions are dynamic
- Targets computed at runtime

---

## Prerequisites

### System Requirements
- Ubuntu 22.04
- ROS2 Jazzy
- Gazebo Harmonic
- TurtleBot3 packages

### Install System Dependencies
```bash
sudo apt install ros-jazzy-turtlebot3* ros-jazzy-ros-gz* python3-pip python3-venv
```

### Create Python Virtual Environment
```bash
python3 -m venv ~/rl_env
source ~/rl_env/bin/activate
pip install stable-baselines3 gymnasium numpy torch tensorboard
```

---

## Setup & Run

### Step 0 — Kill All Old Processes
```bash
pkill -f gzserver
pkill -f gzclient
pkill -f gz
pkill -f ros2
sleep 2
```

### Step 1 — Build ROS2 Package
```bash
cd ~/ros2_ws
colcon build --packages-select tb3_multi_robot
source install/setup.bash
```

### Step 2 — Restart Script
```bash
chmod +x ~/restart_sim.sh
```

### Step 3 — Train RL Model
```bash
python3 train_rl.py
```

### Step 4 — Launch Simulation
```bash
~/restart_sim.sh
```

### Step 5 — Change Robot Positions (optional)
(edit robots.yaml)

### Step 6 — Move Box (optional)
```bash
gz service -s /world/default/set_pose ...
```

### Step 7 — Run Controller
```bash
python3 box_lift_rl_controller.py
```

---

## Expected Output
BOX LIFTED! Approaching box...
All robots touching box!

---

## Debug Commands
```bash
ros2 topic list
ros2 node list
gz model -m lift_box -p
```

---

## How It Works

### Phase 1 — RL Navigation
- Uses PPO
- Reads Gazebo positions
- Computes targets

### Phase 2 — Physical Approach
- Robots move to box
- Task completes

---

## Key Files

| File | Purpose |
|---|---|
| train_rl.py | Training |
| box_lift_rl_controller.py | Control |
| robots.yaml | Config |
| tb3_world.world | Simulation |
| restart_sim.sh | Restart |

---

## Extra Commands

### Kill Processes
```bash
pkill -f gzserver
pkill -f gzclient
pkill -f gz
pkill -f ros2
```

### Change Robot Positions
(edit robots.yaml)

### Run Simulation
```bash
~/restart_sim.sh
```

### Move Box
```bash
gz service ...
```

### Run RL
```bash
python3 box_lift_rl_controller.py
```
