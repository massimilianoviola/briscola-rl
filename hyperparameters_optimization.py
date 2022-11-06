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


def objective(trial):
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    recurrent_layer_size = 256#trial.suggest_int("recurrent_layer_size", 64, 256)
    fully_connected_size = 256#trial.suggest_int("fully_connected_size", 64, 256)
    lr = trial.suggest_loguniform("lr", 1e-8, 1e-1)
    replace_every = 1000 #trial.suggest_int("replace_every", 100, 2000)

    # Initialize agents
    agents = []
    agent = RecurrentDeepQAgent(
        n_features=25,
        n_actions=3,
        epsilon=1.0,
        minimum_epsilon=0.1,
        replay_memory_capacity=1000000,
        minimum_training_samples=2000,
        batch_size=256,
        discount=0.95,
        loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=lr,
        replace_every=replace_every,
        epsilon_decay_rate=0.999998,
        hidden_size=recurrent_layer_size,
        fully_connected_layers=fully_connected_size,
    )

    agents.append(agent)
    agents.append(RandomAgent())

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    num_epochs = 1000
    evaluate_every = 100
    num_evaluations = 100
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
            for agent in agents:
                agent.restore_epsilon()

        # Update target network for Deep Q-learning agent
        if epoch % agents[0].replace_every == 0:
            agents[0].target_net.load_state_dict(
                agents[0].policy_net.state_dict(),
            )
    return current_winrate


study = optuna.create_study(study_name="DRQN", direction="maximize")
study.optimize(objective, n_trials=10)

print("BEST TRIAL:")
trial = study.best_trial
print(f"VALUE: {trial.value}")

print("PARAMS:")
for key, value in trial.params.items():
    print(f"{key}: {value}")
