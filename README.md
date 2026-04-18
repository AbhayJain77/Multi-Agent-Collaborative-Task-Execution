# tb3_box_lift — Multi-Agent RL Collaborative Box Lifting

Three TurtleBot3 robots learn to collaboratively position themselves around a box using **Reinforcement Learning (PPO)**. Each robot has limited upward force — the box lifts only when all 3 reach correct positions simultaneously. The RL model is trained in a custom Gymnasium environment and deployed on real Gazebo simulation via ROS2.

---

## Project Structure

```
tb3_box_lift/
├── multi_robot_scripts/
│   ├── train_rl.py                ← PPO training script (Stable-Baselines3)
│   └── box_lift_rl_controller.py  ← ROS2 node (loads trained model + controls robots)
├── launch/
│   └── tb3_world.launch.py        ← Gazebo world + robot spawner
├── config/
│   └── robots.yaml                ← Robot spawn positions
├── worlds/
│   └── tb3_world.world            ← Gazebo world with box
└── restart_sim.sh                 ← Clean restart script
```
---

## RL Design

| Component      | Detail                                                              |
| -------------- | ------------------------------------------------------------------- |
| Algorithm      | PPO (Proximal Policy Optimization)                                  |
| Observation    | Relative vector `[rel_x, rel_y]` per robot to its target = 6 values |
| Action         | `MultiDiscrete [5,5,5]` — stay/±x/±y per robot                      |
| Reward         | `-distance × 0.1` per step + `+500` terminal bonus when all 3 lift  |
| Target layout  | Equilateral triangle (r=0.6m) around box                            |
| Training steps | 500,000 timesteps                                                   |
| Success rate   | 10/10 in evaluation                                                 |

---

## Prerequisites

### System Requirements

* Ubuntu 22.04
* ROS2 Jazzy
* Gazebo Harmonic
* TurtleBot3 packages

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

### Step 0 — Kill All Old Processes (recommended before every run)

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

### Step 2 — Create Restart Script

```bash
cat > ~/restart_sim.sh << 'EOF'
#!/bin/bash
echo "Killing all processes..."
pkill -f gzserver
pkill -f gzclient
pkill -f gz
pkill -f ros2
pkill -f box_lift
sleep 3
echo "Launching fresh simulation..."
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch tb3_multi_robot tb3_world.launch.py
EOF
chmod +x ~/restart_sim.sh
```

### Step 3 — Train RL Model (only needed once)

```bash
source ~/rl_env/bin/activate
cd ~/ros2_ws/src/tb3_multi_robot/multi_robot_scripts
python3 train_rl.py
```

Training takes ~10 minutes on CPU. Model saved as `box_lift_final.zip`.

Monitor training:

```bash
tensorboard --logdir ./logs/
```

### Step 4 — Launch Gazebo Simulation (mandatory)

```bash
cd ~/ros2_ws
colcon build --packages-select tb3_multi_robot
source install/setup.bash
~/restart_sim.sh
```

### Step 5 — (Optional) Change Robot Spawn Positions

```bash
cat > ~/ros2_ws/src/tb3_multi_robot/config/robots.yaml << 'EOF'
robots:
  - name: tb1
    x_pose: "-4.0"
    y_pose: "-6.0"
    z_pose: 0.01
    enabled: true
  - name: tb2
    x_pose: "-6.0"
    y_pose: "-3.0"
    z_pose: 0.01
    enabled: true
  - name: tb3
    x_pose: "-4.0"
    y_pose: "8.0"
    z_pose: 0.01
    enabled: true
EOF
```

Change x,y coordinates to any values. Then rebuild and relaunch (Step 4).

### Step 6 — (Optional) Move Box to Any Position

```bash
gz service -s /world/default/set_pose \
  --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean \
  --timeout 2000 \
  --req 'name: "lift_box", position: {x: 4.0, y: 7.9, z: 0.075}'
```

Change x,y to any position. Robots will automatically navigate to surround the new box position.

### Step 7 — Run RL Controller (mandatory)

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/rl_env/bin/activate
cd ~/ros2_ws/src/tb3_multi_robot/multi_robot_scripts
python3 box_lift_rl_controller.py
```

---
## Expected Output

```
Box position: [5. 5.]
tb1 spawn:[-5.  2.]  target:[5.  5.6]
tb2 spawn:[4. 3.]    target:[4.4 4.7]
tb3 spawn:[1. -5.]   target:[5.6 4.7]
...
box:[5. 5.]  tb1:6.83  tb2:7.21  tb3:5.44  rl:[1 3 4]
box:[5. 5.]  tb1:5.21  tb2:5.87  tb3:4.12  rl:[1 3 4]
...
box:[5. 5.]  tb1:0.47  tb2:0.44  tb3:0.49  rl:[1 3 2]
BOX LIFTED! Approaching box...
All robots touching box!
```

---

## Debug Commands

```bash
# Check active topics
ros2 topic list

# Monitor robot odometry
ros2 topic echo /tb1/odom

# Monitor velocity commands
ros2 topic echo /tb1/cmd_vel

# Check running nodes
ros2 node list

# Check box position in Gazebo
gz model -m lift_box -p

# Check robot positions in Gazebo
gz model -m tb1_burger -p
gz model -m tb2_burger -p
gz model -m tb3_burger -p
```

---

## How It Works

### Phase 1 — RL Navigation

* RL model reads real box position from Gazebo
* Computes target positions in triangle around box
* Each robot receives `[rel_x, rel_y]` observation — vector to its target
* PPO model outputs discrete action (stay/±x/±y) every 0.1 seconds
* Proportional controller converts discrete action to wheel velocity

### Phase 2 — Physical Approach

* Once all 3 robots within threshold distance of targets → `BOX LIFTED`
* Robots then drive directly into box to physically touch it
* Simulation completes when all robots touch box

---

## Key Files

| File                        | Purpose                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `train_rl.py`               | Train PPO model in custom Gymnasium environment                      |
| `box_lift_rl_controller.py` | ROS2 node — loads model, reads Gazebo positions, controls robots     |
| `robots.yaml`               | Robot spawn positions — change for different starting configurations |
| `tb3_world.world`           | Gazebo world — contains box model                                    |
| `restart_sim.sh`            | Clean restart script — kills all old processes before relaunch       |

---

## Proof That Solution Is NOT Hardcoded (Manual Verification Steps)

*(follow these steps in sequence only)*

### 0) Kill all processes (recommended)

```bash
pkill -f gzserver
pkill -f gzclient  
pkill -f gz
pkill -f ros2
sleep 2
```

### 1) Change robot positions (optional)

```bash
cat > ~/ros2_ws/src/tb3_multi_robot/config/robots.yaml << 'EOF'
robots:
  - name: tb1
    x_pose: "-4.0"
    y_pose: "-6.0"
    z_pose: 0.01
    enabled: true
  - name: tb2
    x_pose: "-6.0"
    y_pose: "-3.0"
    z_pose: 0.01
    enabled: true
  - name: tb3
    x_pose: "-4.0"
    y_pose: "8.0"
    z_pose: 0.01
    enabled: true
EOF
```

Change x,y coordinates of your choice.

---

### 2) Launch simulation (mandatory)

```bash
cd ~/ros2_ws
colcon build --packages-select tb3_multi_robot
source install/setup.bash
~/restart_sim.sh
```

---

### 3) Move box (optional)

```bash
gz service -s /world/default/set_pose \
--reqtype gz.msgs.Pose \
--reptype gz.msgs.Boolean \
--timeout 2000 \
--req 'name: "lift_box", position: {x: 4.0, y: 7.9, z: 0.075}'
```

Change x,y coordinates of your choice.

---

### 4) Run RL controller (mandatory)

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/Desktop/varun/rl_env/bin/activate
cd ~/Desktop/varun/tb3_box_lift/multi_robot_scripts
python3 box_lift_rl_controller.py
```
