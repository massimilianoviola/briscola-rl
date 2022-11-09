import argparse
import pickle
import random
import numpy as np
import torch
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.recurrent_q_agent import RecurrentDeepQAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger


def train():
    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []
    agent = RecurrentDeepQAgent(
        n_features=25,
        n_actions=3,
        epsilon=1.0,
        minimum_epsilon=0.1,
        replay_memory_capacity=1000000,
        minimum_training_samples=100,
        batch_size=32,
        discount=0.95,
        loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=0.01,
        replace_every=500,
        epsilon_decay_rate=0.999998,
        hidden_size=256,
        fully_connected_layers=256,
        optimizer=torch.optim.Adam
    )

    agents.append(agent)
    agents.append(RandomAgent())

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    num_epochs = 25000
    evaluate_every = 1000
    num_evaluations = 1000
    for epoch in range(1, num_epochs + 1):
        print(f"Episode: {epoch} epsilon: {agents[0].epsilon:.6f}", end="\r")

        game_winner_id, winner_points, episode_rewards_log = brisc.play_episode(
            game,
            agents,
        )
        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)

            for agent in agents:
                agent.restore_epsilon()

        # Update target network for Deep Q-learning agent
        if epoch % agents[0].replace_every == 0:
            agents[0].target_net.load_state_dict(
                agents[0].policy_net.state_dict(),
            )


if __name__ == "__main__":
    train()
