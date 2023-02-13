import torch
import numpy as np
import random
import optuna
from agents.recurrent_q_agent import RecurrentDeepQAgent
from agents.random_agent import RandomAgent
from agents.ppo_shared import PPOAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger
import environment as brisc
import torch.optim as optim


def objective(trial):
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    ppo_clip = trial.suggest_float("ppo_clip", .1, .3)
    ppo_steps = trial.suggest_int("ppo_steps", 1, 50)
    value_coeff = trial.suggest_float("value_coeff", .5, 1.0)
    batch_size = 100 # trial.suggest_int("batch_size", 1, 100)
    discount = .90 # trial.suggest_float("discount", .8, .99)

    # Initialize agents
    agents = []
    agent = PPOAgent(
        n_features=65,
        n_actions=3,
        discount=discount,
        critic_loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=lr,
        actor_layers=[256, 256],
        critic_layers=[256, 256],
        ppo_steps=ppo_steps,
        ppo_clip=ppo_clip,
        ent_coeff=0.01,
        value_coeff=value_coeff,
        batch_size=batch_size,
        log=False,
    )
    agents.append(agent)
    agents.append(RandomAgent())

    num_epochs = 10000
    evaluate_every = 1000
    num_evaluations = 1000

    winrates = []
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
            current_winrate = total_wins[0] / (total_wins[0] + total_wins[1])
            winrates.append(current_winrate)
            for agent in agents:
                agent.restore_epsilon()

    return max(winrates)


study = optuna.create_study(study_name="PPO", direction="maximize")
study.optimize(objective, n_trials=100)

print("BEST TRIAL:")
trial = study.best_trial
print(f"VALUE: {trial.value}")

print("PARAMS:")
for key, value in trial.params.items():
    print(f"{key}: {value}")
