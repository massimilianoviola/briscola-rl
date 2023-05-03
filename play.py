import argparse
import environment as brisc
from agents.q_agent import QAgent
from agents.human_agent import HumanAgent
from agents.ai_agent import AIAgent
from utils import BriscolaLogger
import torch


def main(argv=None):
    """Play against one of the intelligent agents."""
    # initialize the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
    game = brisc.BriscolaGame(2, logger)

    # initialize the agents
    agents = []

    if FLAGS.model_dir:
        checkpoint = torch.load(FLAGS.model_dir)
        config = checkpoint['config']
        agent = QAgent(
            n_features=config['n_features'],
            n_actions=config['n_actions'],
            epsilon=config['epsilon'],
            minimum_epsilon=config['minimum_epsilon'],
            replay_memory_capacity=1000000,
            minimum_training_samples=2000,
            batch_size=256,
            discount=0.95,
            loss_fn=torch.nn.SmoothL1Loss(),
            learning_rate=0.0001,
            replace_every=1000,
            epsilon_decay_rate=0.99998,
            layers=config['layers'],
        )
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        print(agent.deck)
        #agent.load(FLAGS.model_dir)
        agent.make_greedy()
        agents.append(agent)
    else:
        agents.append(AIAgent())

    agents.append(HumanAgent())

    brisc.play_episode(game, agents)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        help="Trained model path if you want to play against a deep agent",
        type=str,
    )

    FLAGS = parser.parse_args()

    main()
