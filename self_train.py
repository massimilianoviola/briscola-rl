import argparse
import os
import time
import random
import shutil
import graphic_visualizations as gv
import environment as brisc
from evaluate import evaluate
from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from utils import BriscolaLogger, CardsEncoding, CardsOrder, NetworkTypes, PlayerState


class CopyAgent(QAgent):
    """Copied agent. Identical to a QAgent, but does not update itself."""
    def __init__(self, agent):
        # create a default QAgent
        super().__init__(network=agent.network)

        self.name = 'CopyAgent'

        # make the CopyAgent always greedy since it is not learning
        self.epsilon = 1.0
        self.make_greedy()

        # initialize the CopyAgent with the same weights as the passed QAgent
        if type(agent) is not QAgent:
            raise TypeError("CopyAgent __init__ requires argument of type QAgent")
        # create a temp directory where to save the current agent model
        if not os.path.isdir('__tmp_model_dir__'):
            os.makedirs('__tmp_model_dir__')
        # transfer weights
        agent.save_model('__tmp_model_dir__')
        super().load_model('__tmp_model_dir__')
        # remove the temp directory after loading the model into the CopyAgent
        shutil.rmtree('__tmp_model_dir__')

    def update(self, *args):
        pass


def self_train(game, agent1, num_epochs, evaluate_every, num_evaluations, copy_every, model_dir='', evaluation_dir='evaluation_dir'):
    """Train an agent using self-play, playing games against old copies of itself.
    Performance is frequently evaluated against the random agent and ultimately against the AI agent.
    """
    # initialize the list of old agents with a copy of the non-trained agent
    old_agents = [CopyAgent(agent1)]
    # training starts
    best_total_wins = -1
    for epoch in range(1, num_epochs + 1):
        gv.printProgressBar(epoch, num_epochs,
                            prefix=f"Epoch: {epoch}",
                            length=50)

        # pick an agent from the past as opponent
        agents = [agent1, random.choice(old_agents)]

        # play a game to train the agent
        brisc.play_episode(game, agents)

        # evaluation step
        if epoch % evaluate_every == 0:

            # evaluation visualization directory
            if not os.path.isdir(evaluation_dir):
                os.mkdir(evaluation_dir)

            # make greedy for evaluation
            agent1.make_greedy()

            # evaluation against random agent
            agents = [agent1, RandomAgent()]
            winners, points = evaluate(game, agents, num_evaluations)
            gv.evaluate_summary(winners, points, agents, evaluation_dir +
                "/epoch:" + str(epoch) + " " + agents[0].name + "1 vs " + agents[1].name)
            victory_history_1vR.append(winners)
            points_history_1vR.append(points)
            # save the model if the agent performs better against random agent
            if winners[0] > best_total_wins:
                best_total_wins = winners[0]
                agent1.save_model(model_dir)

            # get ready for more training
            agent1.restore_epsilon()

        if epoch % copy_every == 0:
            # add current agent to the old agents list
            old_agents.append(CopyAgent(agent1))
            # remove the oldest agent if the maximum number of agents is reached
            if len(old_agents) > FLAGS.max_old_agents:
                old_agents.pop(0)

    return best_total_wins


def main(argv=None):

    global victory_history_1vR
    victory_history_1vR  = []

    global points_history_1vR
    points_history_1vR  = []

    # initialize the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # initialize agent
    global agent1
    agent1 = QAgent(
        FLAGS.epsilon,
        FLAGS.epsilon_increment,
        FLAGS.epsilon_max,
        FLAGS.discount,
        FLAGS.network,
        FLAGS.layers,
        FLAGS.learning_rate,
        FLAGS.replace_target_iter,
        FLAGS.batch_size
    )

    # training
    start_time = time.time()
    best_total_wins = self_train(game, agent1,
                                FLAGS.num_epochs,
                                FLAGS.evaluate_every,
                                FLAGS.num_evaluations,
                                FLAGS.copy_every,
                                FLAGS.model_dir)
    print(f"\nBest winning ratio : {best_total_wins/FLAGS.num_evaluations:.2%}")
    print(f"Total time elapsed: {time.time() - start_time:.2f}")

    # summary plots
    x = [FLAGS.evaluate_every * i for i in range(1, 1+len(victory_history_1vR))]

    # 1vRandom
    vict_hist = victory_history_1vR
    point_hist = points_history_1vR
    labels = [agent1.name+'1', RandomAgent().name]
    gv.training_summary(x, vict_hist, point_hist, labels, FLAGS, "evaluation_dir/1vR")

    # evaluate against AI agent
    agents = [agent1, AIAgent()]
    winners, points = evaluate(game, agents, FLAGS.num_evaluations)
    gv.evaluate_summary(winners, points, agents, "evaluation_dir/"+
        agents[0].name + "1 vs " + agents[1].name)


if __name__ == '__main__':

    # parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument("--model_dir", default="saved_model", help="Where to save the trained model, checkpoints and stats", type=str)
    parser.add_argument("--num_epochs", default=1000, help="Number of training games played", type=int)
    parser.add_argument("--max_old_agents", default=50, help="Maximum number of old copies of QAgent stored", type=int)
    parser.add_argument("--copy_every", default=100, help="Add the copy after tot number of epochs", type=int)

    # evaluation parameters
    parser.add_argument("--evaluate_every", default=100, help="Evaluate model after this many epochs", type=int)
    parser.add_argument("--num_evaluations", default=500, help="Number of evaluation games against each type of opponent for each test", type=int)

    # state parameters
    parser.add_argument("--cards_order", default=CardsOrder.APPEND, choices=[CardsOrder.APPEND, CardsOrder.REPLACE, CardsOrder.VALUE], help="Where a drawn card is put in the hand")
    parser.add_argument("--cards_encoding", default=CardsEncoding.HOT_ON_NUM_SEED, choices=[CardsEncoding.HOT_ON_DECK, CardsEncoding.HOT_ON_NUM_SEED], help="How to encode cards")
    parser.add_argument("--player_state", default=PlayerState.HAND_PLAYED_BRISCOLA, choices=[PlayerState.HAND_PLAYED_BRISCOLA, PlayerState.HAND_PLAYED_BRISCOLASEED, PlayerState.HAND_PLAYED_BRISCOLA_HISTORY], help="Which cards to encode in the player state")

    # RL parameters
    parser.add_argument("--epsilon", default=0, help="How likely is the agent to choose the best reward action over a random one", type=float)
    parser.add_argument("--epsilon_increment", default=5e-5, help="How much epsilon is increased after each action taken up to epsilon_max", type=float)
    parser.add_argument("--epsilon_max", default=0.85, help="The maximum value for the incremented epsilon", type=float)
    parser.add_argument("--discount", default=0.85, help="How much a reward is discounted after each step", type=float)

    # network parameters
    parser.add_argument("--network", default=NetworkTypes.DQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN], help="Neural Network used for approximating value function")
    parser.add_argument('--layers', default=[256, 128], help="Definition of layers for the chosen network", type=int, nargs='+')
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for the network updates", type=float)
    parser.add_argument("--replace_target_iter", default=2000, help="Number of update steps before copying evaluation weights into target network", type=int)
    parser.add_argument("--batch_size", default=100, help="Training batch size", type=int)


    FLAGS = parser.parse_args()

    main()