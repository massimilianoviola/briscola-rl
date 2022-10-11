import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from networks.dqn import ReplayMemory
import torch
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger
from utils import CardsEncoding, CardsOrder, NetworkTypes, PlayerState

def train(game, agents, num_epochs, evaluate_every, num_evaluations, save=False, save_dir = ""):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    best_total_wins = -1
    rewards_per_episode = []
    winrates = []
    for epoch in range(1, num_epochs + 1):
        print (f"Episode: {epoch} epsilon: {agents[0].epsilon:.6f} replay_memory_size: {len(agents[0].replay_memory)}", end='\r')

        game_winner_id, winner_points, episode_rewards_log = brisc.play_episode(game, agents)

        rewards_per_episode.append(episode_rewards_log)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)
            winrates.append(total_wins)
            sns.set_theme()
            for agent in agents:
                agent.restore_epsilon()
            if total_wins[0] > best_total_wins:
                best_total_wins = total_wins[0]
        
        if epoch % agents[0].replace_every == 0:
            #print("UPDATING TARGET")
            agents[0].target_net.load_state_dict(agents[0].policy_net.state_dict())

    if save:
        with open(save_dir + "/rewards.pkl", "wb") as f:
            pickle.dump(rewards_per_episode, f)
        
        with open(save_dir + "/winrates.pkl", "wb") as f:
            pickle.dump(winrates, f) 
    
    return best_total_wins, rewards_per_episode



def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Agent to train", default="QLearningAgent")
    parser.add_argument("--episodes", type=int, help="Number of training episodes", default=10000)
    parser.add_argument("--epsilon", type=float, help="Starting value of epsilon", default=1.0)
    parser.add_argument("--minimum_epsilon", type=float, help="Final value of epsilon", default=0.1)
    parser.add_argument("--epsilon_decay_rate", type=float, help="Epsilon decay rate", default=0.999998)
    parser.add_argument("--discount", type=float, help="Discount factor", default=0.95)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--evaluate_every", type=int, help="Number of episode after which evaluate the agent", default=1000)
    parser.add_argument("--num_evaluation", type=int, help="Number of games to perform evaluation", default=1000)
    parser.add_argument("--replace_every", type=int, help="", default=1000)
    parser.add_argument("--against", type=str, help="Agent to train against", default="RandomAgent")

    args = parser.parse_args()

    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []
    agent = QAgent(
        n_features=25,
        n_actions=3,
        epsilon=args.epsilon,
        minimum_epsilon=args.minimum_epsilon,
        replay_memory_capacity=1000000,
        minimum_training_samples=2000,
        batch_size=256, #256
        discount=args.discount,
        loss_fn=torch.nn.SmoothL1Loss(),
        learning_rate=args.lr,
        replace_every=args.replace_every, #25
        epsilon_decay_rate=args.epsilon_decay_rate,
        layers= [256, 256] #  [128,128] [64,32]
    )
    agents.append(agent)
    if args.against == "AIAgent":
        agent = AIAgent() 
    else:
        agent = RandomAgent() 
    agents.append(agent)

    _, rewards_per_episode = train(game, agents, args.episodes, args.evaluate_every, args.num_evaluation, False, "./results")

    print("FINISHED")

if __name__ == "__main__":
    main()

