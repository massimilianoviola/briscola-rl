import os
import argparse
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import environment as brisc
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger


def train(
        game,
        agents,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        save: bool = False,
        save_dir: str = "",
        checkpoint=None,
):
    """The agent is trained for num_epochs number of episodes following an
    epsilon-greedy policy. Every evaluate_every number of episodes the agent
    is evaluated by playing num_evaluations number of games.
    The winrate is obtained from these evaluations is used to select the best
    model and its weights are saved.
    """

    best_total_wins = -1
    best_winrate = 0.0

    rewards_per_episode = checkpoint['rewards']
    print(f"N° actual rewards: {len(rewards_per_episode)}")
    points_log = checkpoint['points']
    winrates = checkpoint['winrates']

    for e in winrates:
        winrate = e[0] / (e[0] + e[1])
        if winrate > best_winrate:
            best_winrate = winrate
    print(f"Best winrate: {best_winrate * 100}%")

    if save:
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        game_winner_id, winner_points, episode_rewards_log = brisc.play_episode(
            game,
            agents,
            train=True,
        )

        rewards_per_episode.append(episode_rewards_log)
        if agents[game_winner_id].name == "QlearningAgent":
            points_log.append(winner_points)
        else:
            points_log.append(120 - winner_points)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)
            for agent in agents:
                agent.restore_epsilon()
            winrates.append(total_wins)

            current_winrate = total_wins[0] / (total_wins[0] + total_wins[1])
            if current_winrate > best_winrate and save:
                print(f"Saving the checkpoint...\n"
                      f"New best winrate: {round(current_winrate * 100, 2)}% (Previous: {best_winrate * 100}%)")
                best_winrate = current_winrate

                checkpoint['config'] = vars(agents[0])
                checkpoint['policy_state_dict'] = agents[0].policy_net.state_dict()
                checkpoint['optimizer_state_dict'] = agents[0].optimizer.state_dict()
                checkpoint['rewards'] = rewards_per_episode
                checkpoint['winrates'] = winrates
                checkpoint['points'] = points_log

                torch.save(checkpoint, save_dir)
                # agents[0].save(save_dir + "model.pt")
                print(f"New # rewards: {len(checkpoint['rewards'])}\n"
                      f"Actual epsilon: {round(agents[0].epsilon, 4)}\n"
                      f"Checkpoint SAVED...\n")

            if total_wins[0] > best_total_wins:
                best_total_wins = total_wins[0]

        # Update target network for Deep Q-learning agent
        if epoch % agents[0].replace_every == 0:
            agents[0].target_net.load_state_dict(
                agents[0].policy_net.state_dict(),
            )

        print(f"Episode: {epoch} epsilon: {agents[0].epsilon:.4f}", end="\r")
    end_time = time.time()
    print("\nComputation time: {:.2f} seconds".format(end_time - start_time))
    # if save:
    #     with open(save_dir + "rewards.pkl", "wb") as f:
    #         pickle.dump(rewards_per_episode, f)
    #
    #     with open(save_dir + "winrates.pkl", "wb") as f:
    #         pickle.dump(winrates, f)
    #
    #     with open(save_dir + "points.pkl", "wb") as f:
    #         pickle.dump(points_log, f)
    return best_total_wins, rewards_per_episode


def main(argv=None):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        help="Agent to train",
        default="QLearningAgent",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        help="Starting value of epsilon",
        default=1.0,
    )

    parser.add_argument(
        "--minimum_epsilon",
        type=float,
        help="Final value of epsilon",
        default=0.1,
    )

    parser.add_argument(
        "--epsilon_decay_rate",
        type=float,
        help="Epsilon decay rate",
        default=0.999998,
    )

    parser.add_argument(
        "--discount",
        type=float,
        help="Discount factor",
        default=0.95,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1e-4,
    )

    # ----------------------------
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes",
        default=25000,
    )

    parser.add_argument(
        "--evaluate_every",
        type=int,
        help="Number of episode after which evaluate the agent",
        default=1000,
    )
    # ----------------------------

    parser.add_argument(
        "--num_evaluation",
        type=int,
        help="Number of games to perform evaluation",
        default=200,
    )

    parser.add_argument(
        "--replace_every",
        type=int,
        help="",
        default=1000,
    )

    parser.add_argument(
        "--against",
        type=str,
        help="Agent to train against",
        default="RandomAgent",
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path where model/data is saved.",
        default=f"models/{parser.parse_args().agent}.pt",
    )

    parser.add_argument(
        "--winning_reward",
        type=int,
        help="Extra reward given for winning the game",
        default=100,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path of the model checkpoint",
        default=f"models/{parser.parse_args().agent}.pt",
    )
    args = parser.parse_args()

    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger, gui_obj=None, bonus=args.winning_reward)

    # Initialize agents
    agents = []
    agent = QAgent(
        n_features=27,
        n_actions=3,
        epsilon=args.epsilon,
        minimum_epsilon=args.minimum_epsilon,
        replay_memory_capacity=1000000,
        minimum_training_samples=2000,
        batch_size=256,
        discount=args.discount,
        loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=args.lr,
        replace_every=args.replace_every,
        epsilon_decay_rate=args.epsilon_decay_rate,
        layers=[256, 256],
        state_type=1,
    )
    checkpoint = {
        'config': vars(agent),
        'info': 'get_state, reward for winning, vs RulesAgent',
        'policy_state_dict': None,
        'optimizer_state_dict': None,
        'rewards': [],
        'winrates': [],
        'points': [],
    }
    if args.checkpoint_path:
        print("Resuming training from checkpoint...\n")
        try:
            checkpoint = torch.load(args.checkpoint_path)
            config = checkpoint['config']
            agent = QAgent(
                n_features=config['n_features'],
                n_actions=config['n_actions'],
                epsilon=config['epsilon'],
                minimum_epsilon=config['minimum_epsilon'],
                replay_memory_capacity=1000000,
                minimum_training_samples=config['minimum_training_samples'],
                batch_size=config['batch_size'],
                discount=config['discount'],
                loss_fn=config['loss_fn'],
                learning_rate=0.0001,
                replace_every=config['replace_every'],
                epsilon_decay_rate=config['epsilon_decay_rate'],
                layers=config['layers'],
            )
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        except FileNotFoundError:
            print("ERROR: the checkpoint file does not exist. Check the arguments specified in the file train.py."
                  "\nTraining a new model...\n")
            pass

    agents.append(agent)
    if args.against == "AIAgent":
        agent = AIAgent()
    else:
        agent = RandomAgent()
    agents.append(agent)

    save_model = True if args.path else False

    _, rewards_per_episode = train(
        game,
        agents,
        args.episodes,
        args.evaluate_every,
        args.num_evaluation,
        save=save_model,
        save_dir=args.path,
        checkpoint=checkpoint,
    )

    print("\nFINISHED TRAINING")


if __name__ == "__main__":
    main()
