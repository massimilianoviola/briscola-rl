import argparse
from statistics import mean

import torch
from tqdm import tqdm

from agents.random_agent import RandomAgent
from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from graphic_visualizations import stats_plotter
import environment as brisc
from utils import BriscolaLogger, NetworkTypes


def evaluate(game, agents, num_evaluations):
    """Play num_evaluations games and report statistics."""
    total_wins = [0] * len(agents)
    points_history = [[] for _ in range(len(agents))]

    for _ in tqdm(range(num_evaluations)):
        game_winner_id, winner_points, _ = brisc.play_episode(game, agents, train=False)
        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id:
                total_wins[player.id] += 1

    print(f"\nTotal wins: {total_wins}.")
    for i in range(len(agents)):
        print(
            f"{agents[i].name} {i} won {total_wins[i] / num_evaluations:.2%} with an average of {mean(points_history[i]):.2f} points.")

    return total_wins, points_history


def main(args=None):
    """Evaluate agent performance against RandomAgent and AIAgent."""
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = brisc.BriscolaGame(2, logger)

    # agent to be evaluated is RandomAgent or QAgent if a model is provided
    if FLAGS.model_dir:
        print(f"Loading the model '{args.model_dir}'...")
        checkpoint = torch.load(args.model_dir)
        config = checkpoint['config']
        agent = QAgent(
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
            state_type=config['state_type'],
        )
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.make_greedy()
    else:
        agent = RandomAgent()

    # test agent against RandomAgent
    print(f"Testing against RandomAgent on {args.num_evaluations} games")
    agents = [agent, RandomAgent()]
    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)

    # test agent against AIAgent
    print(f"Testing against AIAgent on {args.num_evaluations} games")
    agents = [agent, AIAgent()]
    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)


if __name__ == '__main__':
    # parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="models/QLearningAgent_25k_66.7%_ruled.pt",
                        help="Provide a trained model path if you want to play against a deep agent", type=str)
    parser.add_argument("--network", default=NetworkTypes.DQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN],
                        help="Neural network used for approximating value function")
    parser.add_argument("--num_evaluations", default=10000,
                        help="Number of evaluation games against each type of opponent for each test", type=int)

    FLAGS = parser.parse_args()

    main(FLAGS)
