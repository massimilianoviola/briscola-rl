import argparse
import pickle
import random
import numpy as np
import torch
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ppo_shared import PPOAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger


def train():
    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize agents
    agents = []
    agent = PPOAgent(
        n_features=65,
        n_actions=3,
        discount=0.9,
        critic_loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=5e-4,
        actor_layers=[256, 256],
        critic_layers=[256, 256],
        ppo_steps=10,
        ppo_clip=0.1,
        ent_coeff=0.0,
        value_coeff=0.5,
        batch_size=100,
    )

    agents.append(agent)
    agents.append(RandomAgent())
    #agents.append(AIAgent())

    num_epochs = 1000000
    evaluate_every = 1000
    num_evaluations = 1000

    for epoch in range(1, num_epochs + 1):
        print(f"Episode: {epoch}", end="\r")
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


if __name__ == "__main__":
    train()
