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
        agent = QAgent(
            n_features=65,
            n_actions=3,
            epsilon=1.0,
            minimum_epsilon=0.1,
            replay_memory_capacity=1000000,
            minimum_training_samples=2000,
            batch_size=256,
            discount=0.95,
            loss_fn=torch.nn.SmoothL1Loss(),
            learning_rate=0.0001,
            replace_every=1000,
            epsilon_decay_rate=0.99998,
            layers=[256, 256],
        )

        agent.load(FLAGS.model_dir)
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
