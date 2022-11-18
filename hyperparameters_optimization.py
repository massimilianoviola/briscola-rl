import torch
import numpy as np
import random
import optuna
from agents.recurrent_q_agent import RecurrentDeepQAgent
from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger
import environment as brisc
import torch.optim as optim


def objective(trial):
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    recurrent_layer_size = 64

    fully_connected_size = 64

    optimizer = "RMSprop"

    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    momentum = trial.suggest_float("momentum", 0.9, 1.0)
    replace_every = 500

    if optimizer == "SGD":
        opt = optim.SGD
    elif optimizer == "RMSprop":
        opt = optim.RMSprop
    else:
        opt = optim.Adam

    # Initialize agents
    agents = []
    agent = RecurrentDeepQAgent(
        n_features=25,
        n_actions=3,
        epsilon=1.0,
        minimum_epsilon=0.1,
        replay_memory_capacity=1000000,
        minimum_training_samples=500,
        batch_size=32,
        discount=0.95,
        loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=lr,
        replace_every=replace_every,
        epsilon_decay_rate=0.999998,
        hidden_size=recurrent_layer_size,
        fully_connected_layers=fully_connected_size,
        optimizer=opt,
        momentum=momentum,
        sequence_len=4,
    )

    agents.append(agent)
    agents.append(RandomAgent())

    num_epochs = 2000
    evaluate_every = 500
    num_evaluations = 1000

    winrates = []
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
            current_winrate = total_wins[0] / (total_wins[0] + total_wins[1])
            winrates.append(current_winrate)
            for agent in agents:
                agent.restore_epsilon()

        # Update target network for Deep Q-learning agent
        if epoch % agents[0].replace_every == 0:
            agents[0].target_net.load_state_dict(
                agents[0].policy_net.state_dict(),
            )
    return max(winrates)


study = optuna.create_study(study_name="DRQN", direction="maximize")
study.optimize(objective, n_trials=100)

print("BEST TRIAL:")
trial = study.best_trial
print(f"VALUE: {trial.value}")

print("PARAMS:")
for key, value in trial.params.items():
    print(f"{key}: {value}")
