import argparse
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ppo_separated import PPOAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger


def train():
    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger, 0)

    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize agents
    agents = []
    agent = PPOAgent(
        n_features=26,
        n_actions=3,
        discount=0.90,
        gae_lambda=0.90,
        critic_loss_fn=torch.nn.SmoothL1Loss(),
        #learning_rate=5e-4,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        actor_layers=[128, 128],
        critic_layers=[128, 128],
        ppo_steps=10,
        ppo_clip=0.2,
        ent_coeff=0.03,
        #value_coeff=0.5,
        batch_size=64,
        log=True,
    )

    #separated 69.70% and 69.98 points entr -0.07 at 185k episodes
    #shared 76.00% and 72.93 points entr -0.29 at 235k episodes
    agents.append(agent)
    agents.append(RandomAgent())
    #agents.append(AIAgent())

    num_epochs = 200000
    evaluate_every = 1000 #*agents[0].batch_size
    num_evaluations = 1000
    y = []
    x = list(range(num_epochs))
    for epoch in range(1, num_epochs + 1):
        print(f"EPISODE: {epoch}", end="\r")
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
        #print(f"RETURN: {sum(episode_rewards_log['PPOAgent'])}")
        #y.append(sum(episode_rewards_log['PPOAgent']))
    #plt.plot(x, y)
    #plt.show()

if __name__ == "__main__":
    train()
