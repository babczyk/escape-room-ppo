# Proximity Policy Optimization (PPO) Escape Room Solver

This project demonstrates the use of **Proximity Policy Optimization (PPO)**, a reinforcement learning algorithm, to solve an escape room simulation. The agent navigates a room with a **pickable box**, a **button** that opens a door, and **walls**. This implementation is written in **C#** using **MonoGame** for the environment simulation and the PPO algorithm to train the agent.

## Table of Contents
1. [Overview](#overview)
2. [Reinforcement Learning Fundamentals](#reinforcement-learning-fundamentals)
3. [Proximity Policy Optimization (PPO)](#proximity-policy-optimization-ppo)
4. [Project Setup](#project-setup)
5. [How It Works](#how-it-works)
6. [Results and Performance](#results-and-performance)
7. [Future Improvements](#future-improvements)
8. [License](#license)

## Overview
This project involves training an agent to solve an **escape room** using **Proximity Policy Optimization (PPO)**, a reinforcement learning algorithm. The escape room features:
- A **pickable box**: The agent can interact with it to earn rewards.
- A **button**: Pressing the button opens the door to the next room or goal.
- **Walls**: Obstacles the agent must navigate around.
- **Doors**: The exit that the agent must open to complete the task.

The agent uses **PPO** to learn an optimal policy through trial and error, adjusting its actions to maximize cumulative rewards.

## Reinforcement Learning Fundamentals
Reinforcement Learning (RL) is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment**. The agent receives feedback in the form of **rewards** and **penalties**, which it uses to adjust its actions over time.

Key Components:
- **Agent**: The entity that learns from the environment (in this case, the escape room solver).
- **Environment**: The surroundings in which the agent operates (the escape room).
- **Reward**: A numerical value reflecting the desirability of an action (e.g., a positive reward for pressing the button).

The agent uses an algorithm like **PPO** to optimize its actions, learning to make decisions that lead to the highest rewards.

## Proximity Policy Optimization (PPO)
**Proximity Policy Optimization (PPO)** is a reinforcement learning algorithm used to train an agent. It works by optimizing the **policy**—the set of actions the agent should take given its state in the environment. PPO balances exploration (trying new actions) with exploitation (relying on actions that have been successful in the past).

### PPO Algorithm:
1. **Collect Data**: The agent explores the environment and collects state-action-reward trajectories.
2. **Optimize the Policy**: Using the collected data, PPO updates the agent's policy to improve its decision-making and maximize rewards.
3. **Repeat**: The agent continues to explore and learn until it can efficiently solve the escape room.

In this implementation, PPO helps the agent figure out how to interact with the pickable box, press the button, and open the door.

## Project Setup
To run this project, you need the following tools and libraries:
- **C#** (with .NET Core/Framework)
- **MonoGame** (for the escape room simulation)
- **VS Code** (IDE for development)

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/escape-room-ppo.git
   cd escape-room-ppo
   ```

2. Install MonoGame:
   - Follow the installation guide for [MonoGame](https://www.monogame.net/downloads/) to set up MonoGame in your environment.

3. Build the project:
   Open the project in **VS Code** and build it using the following command:
   ```bash
   dotnet build
   ```

4. Run the game:
   After building, run the game simulation:
   ```bash
   dotnet run
   ```

## How It Works
The escape room environment consists of the following components:
- **Pickable Box**: An object the agent can pick up to receive a reward.
- **Button**: When pressed, the button opens the door.
- **Walls**: Obstacles the agent must avoid or navigate around.
- **Doors**: The goal of the room, which can be opened by pressing the button.

The agent makes decisions based on its position relative to the objects in the environment. Using **PPO**, the agent explores the environment and refines its policy to:
1. Pick up the box.
2. Press the button.
3. Open the door.

### Key Code Components:
- **State Representation**: The environment is modeled as a grid where the agent’s position and the objects (box, button, walls, door) are represented.
- **PPO Algorithm**: The PPO algorithm is implemented in the `Agent` class, where the agent learns from interactions with the environment.
- **Action Selection**: The agent chooses actions based on its current state, aiming to maximize the cumulative reward.

## Results and Performance
The PPO agent is evaluated on how efficiently it can solve the escape room. After training over several episodes, the agent should be able to:
1. Pick up the box.
2. Press the button.
3. Open the door and escape the room.

Performance is measured by the number of episodes the agent needs to complete the task and how quickly it learns the optimal strategy.

## Future Improvements
- **Increase Environment Complexity**: Add more objects or obstacles to the room to make the environment more challenging.
- **Expand Action Space**: Introduce additional actions, such as jumping or interacting with multiple objects.
- **Hyperparameter Tuning**: Experiment with different PPO hyperparameters to improve the agent's performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

