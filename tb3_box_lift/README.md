# tb3_box_lift — Multi-Agent RL Collaborative Box Lifting

Three TurtleBot3 robots learn to position themselves at correct spots around a box
so their combined upward force lifts it. Positions are discovered by RL (PPO).

## Project Structure
```
tb3_box_lift/
├── multi_robot_scripts/
│   ├── box_lift_env.py            ← Gymnasium RL environment
│   ├── train_rl.py                ← Training script (PPO, Stable-Baselines3)
│   └── box_lift_rl_controller.py  ← ROS2 node (runs trained model)
├── launch/
│   ├── tb3_world.launch.py        ← Gazebo world + robot spawner
│   └── box_lift_rl.launch.py      ← Full demo launch
├── config/
│   └── robots.yaml                ← 3 robots enabled
```

## Prerequisites
- Ubuntu 22.04, ROS2 Humble, Gazebo 11, TurtleBot3 packages
- Python: `pip install stable-baselines3 gymnasium numpy torch`

## Step 1 — Build ROS2 Package
```bash
cd ~/ros2_ws/src
cp -r /path/to/tb3_box_lift .
cd ~/ros2_ws
colcon build --packages-select tb3_multi_robot
source install/setup.bash
```

## Step 2 — Train RL Model (no ROS needed)
```bash
cd ~/ros2_ws/src/tb3_box_lift/multi_robot_scripts
python3 train_rl.py
```
Monitor with TensorBoard:
```bash
tensorboard --logdir ./logs/
```

## Step 3 — Launch Gazebo
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_multi_robot tb3_world.launch.py
```

## Step 4 — Run RL Controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tb3_multi_robot box_lift_rl_controller
```
Or launch everything at once:
```bash
ros2 launch tb3_multi_robot box_lift_rl.launch.py
```

## Useful Debug Commands
```bash
ros2 topic list
ros2 topic echo /tb1/odom
ros2 topic echo /tb1/cmd_vel
ros2 node list
```

## RL Design
| Component | Detail |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| State | Robot positions x3 + box position + lifted flag |
| Action | MultiDiscrete [5,5,5] per robot |
| Reward | +100 lift, -0.1/step, -5 collision, shaped proximity |
| Target | Equilateral triangle r=0.5m around box |





# Ctrl+C to stop Gazebo, then:
pkill -f gzserver
pkill -f gzclient
pkill -f robot_state_publisher




export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch tb3_multi_robot tb3_world.launch.py



source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/Desktop/varun/rl_env/bin/activate
cd ~/Desktop/varun/tb3_box_lift/multi_robot_scripts
python3 box_lift_rl_controller.py