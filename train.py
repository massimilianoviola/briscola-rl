import random
import numpy as np
import torch
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
from utils import BriscolaLogger
from utils import CardsEncoding, CardsOrder, NetworkTypes, PlayerState

def train(game, agents, num_epochs, evaluate_every, num_evaluations, model_dir = ""):

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    best_total_wins = -1

    for epoch in range(1, num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')

        game_winner_id, winner_points = brisc.play_episode(game, agents)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)
            for agent in agents:
                agent.restore_epsilon()
            if total_wins[0] > best_total_wins:
                best_total_wins = total_wins[0]

    return best_total_wins



def main(argv=None):

    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []
    agent = QAgent(
        n_features=70,
        n_actions=3,
        initial_epsilon=0.5,
        final_epsilon=0.05,
        replay_memory_capacity=10000,
        minimum_training_samples=100,
        batch_size=100,
        discount=0.95,
        loss_fn=torch.nn.MSELoss(),
        learning_rate=1e-4,
        replace_every=2000,
        decay_epsilon_until=9000
    )
    agents.append(agent)
    agent = RandomAgent()
    agents.append(agent)

    train(game, agents, 10000, 1000, 500, "")

if __name__ == "__main__":
    main()

