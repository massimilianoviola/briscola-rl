import os
import argparse
import pickle
import random
import numpy as np
import torch
import environment as brisc
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.ppo_separated import PPOAgent
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
    checkpoint = {},
):
    """The agent is trained for num_epochs number of episodes following an
    epsilon-greedy policy. Every evaluate_every number of episodes the agent
    is evaluated by playing num_evaluations number of games.
    The winrate iobtained from these evaluations is used to select the best
    model and its weights are saved.
    """

    best_total_wins = -1
    best_winrate = 0.0

    rewards_per_episode = []
    points_log = []
    winrates = []
    if save:
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

    for epoch in range(1, num_epochs + 1):
        game_winner_id, winner_points, episode_rewards_log = brisc.play_episode(
            game,
            agents,
        )

        rewards_per_episode.append(episode_rewards_log)
        if agents[game_winner_id].name == "PPOAgent":
            points_log.append(winner_points)
        else:
            points_log.append(120 - winner_points)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)
            winrates.append(total_wins)

            current_winrate = total_wins[0] / (total_wins[0] + total_wins[1])
            if current_winrate > best_winrate and save:
                best_winrate = current_winrate
                checkpoint['actor_state_dict'] = agents[0].actor_net.state_dict()
                checkpoint['critic_state_dict'] = agents[0].critic_net.state_dict()
                checkpoint['actor_optimizer_state_dict'] = agents[0].actor_opt.state_dict()
                checkpoint['critic_optimizer_state_dict'] = agents[0].critic_opt.state_dict()

                torch.save(checkpoint, save_dir)
                # agents[0].save(save_dir + "model.pt")
                print("SAVED")

            for agent in agents:
                agent.restore_epsilon()
            if total_wins[0] > best_total_wins:
                best_total_wins = total_wins[0]


        print(f"Episode: {epoch}", end="\r")

    if save:
        checkpoint['rewards'] = rewards_per_episode
        checkpoint['winrates'] = winrates
        checkpoint['points'] = points_log
        torch.save(checkpoint, save_dir)
    
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
        "--episodes",
        type=int,
        help="Number of training episodes",
        default=10000,
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

    parser.add_argument(
        "--evaluate_every",
        type=int,
        help="Number of episode after which evaluate the agent",
        default=1000,
    )

    parser.add_argument(
        "--num_evaluation",
        type=int,
        help="Number of games to perform evaluation",
        default=1000,
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
    )

    parser.add_argument(
        "--winning_reward",
        type=int,
        help="Extra reward given for winning the game",
        default=0,
    )
    args = parser.parse_args()

    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger, args.winning_reward)

    # Initialize agents
    agents = []
    agent = PPOAgent(
        n_features=26,
        n_actions=3,
        discount=0.90,
        gae_lambda=0.95,
        critic_loss_fn=torch.nn.SmoothL1Loss(),
        #learning_rate=5e-4,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        actor_layers=[128, 128],
        critic_layers=[128, 128],
        ppo_steps=10,
        ppo_clip=0.2,
        ent_coeff=0.05,
        #value_coeff=0.5,
        batch_size=100,
        log=True,
    )

    checkpoint = {
        'config': vars(agent),
        'info': 'reward for winning, vs RulesAgent',
        'actor_state_dict': None,
        'critic_state_dict': None,
        'actor_optimizer_state_dict': None,
        'critic_optimizer_state_dict': None,
        'rewards': [],
        'winrates': [],
        'points': [],
    }

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

    print("FINISHED TRAINING")


if __name__ == "__main__":
    main()
