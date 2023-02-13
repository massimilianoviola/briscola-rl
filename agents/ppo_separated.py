import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from typing import List


class actor(nn.Module):
    """Given a state outputs actions probabilities"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int],
    ) -> None:
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], output_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.MLP(x)


class critic(nn.Module):
    """Given a state outputs its value"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int],
    ) -> None:
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], output_dim),
        )

    def forward(self, x):
        return self.MLP(x)


class PPOAgent:
    """PPO with separated networks to represent policy and value function"""
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        discount: float,
        critic_loss_fn,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_layers: List[int] = [256, 256],
        critic_layers: List[int] = [256, 256],
        ppo_steps: int = 5,
        ppo_clip: float = 0.2,
        ent_coeff: float = 0.0,
        batch_size: int = 32,
        device=None,
    ) -> None:
        self.name = "PPOAgent"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.n_features = n_features
        self.n_actions = n_actions

        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None

        self.states_batch = []
        self.actions_batch = []
        self.log_probs_batch = []
        self.rewards_batch = []
        self.episode_rewards = []

        self.discount = discount
        self.deck = np.zeros((10, 4))

        self.loss_fn = critic_loss_fn
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate

        self.actor_net = actor(n_features, n_actions, actor_layers).to(self.device)
        self.critic_net = critic(n_features, 1, critic_layers).to(self.device)

        self.actor_opt = optim.Adam(
            self.actor_net.parameters(),
            lr=actor_learning_rate,
        )

        self.critic_opt = optim.Adam(
            self.critic_net.parameters(),
            lr=critic_learning_rate,
        )

        self.ppo_steps = ppo_steps
        self.ppo_clip = ppo_clip
        self.ent_coeff = ent_coeff
        self.batch_size = batch_size

    def observe(self, env, player):
        """Observes the environment and updates the state"""
        self.get_state(env, player)

    def get_state(self, env, player):
        """"""
        state = np.zeros(25)
        value_offset = 1
        seed_offset = 3
        features_per_card = 6
        state[0] = player.points
        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + seed_offset + card.seed
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        for i, card in enumerate(env.played_cards):
            number_index = i + 3 * features_per_card + value_offset
            seed_index = i + 3 * features_per_card + seed_offset + card.seed
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        deck = self.deck.flatten()
        state = np.concatenate((state, deck))

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def select_action(self, available_actions):
        state = torch.from_numpy(self.state).float().to(self.device)
        actions_probabilities = self.actor_net(state)
        policy = distributions.Categorical(actions_probabilities)
        takes = 0
        while True:
            takes += 1
            if takes >= 100:
                action = np.random.choice(available_actions)
                self.action = torch.as_tensor(action)
                self.log_prob = policy.log_prob(self.action).detach()
                break
            self.action = policy.sample().detach()
            self.log_prob = policy.log_prob(self.action).detach()
            if self.action in available_actions:
                break

        return self.action.item()

    def get_returns(self, rewards_batch):
        returns = []
        for rewards in reversed(rewards_batch):
            G = 0
            for r in reversed(rewards):
                G = r + self.discount * G
                returns.append(G)

        return returns[::-1]

    def get_values(self, states):
        values = self.critic_net(states).squeeze()
        return values

    def get_advantages(self, returns, values):
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        return adv

    def get_current_log_probs(self, states, actions):
        probabilities = self.actor_net(states)
        policy = distributions.Categorical(probabilities)
        current_log_probs = policy.log_prob(actions)
        return current_log_probs, policy.entropy()

    def update(self, reward):
        self.reward = reward

        self.states_batch.append(self.last_state)
        self.actions_batch.append(self.action)
        self.log_probs_batch.append(self.log_prob)
        self.episode_rewards.append(self.reward)

        if self.done:
            self.rewards_batch.append(self.episode_rewards)

            self.episode_rewards = []
            if len(self.rewards_batch) >= self.batch_size:
                self.learn()

                self.states_batch = []
                self.actions_batch = []
                self.log_probs_batch = []
                self.rewards_batch = []

    def learn(self):
        self.states_batch = torch.Tensor(np.array(self.states_batch))
        self.actions_batch = torch.Tensor(np.array(self.actions_batch))
        self.log_probs_batch = torch.Tensor(np.array(self.log_probs_batch))
        self.rewards_batch = torch.Tensor(np.array(self.rewards_batch))

        returns = self.get_returns(self.rewards_batch)
        returns = torch.Tensor(np.array(returns))
        values = self.get_values(self.states_batch)
        advantages = self.get_advantages(returns, values)

        for _ in range(self.ppo_steps):
            values = self.get_values(self.states_batch)
            curr_log_probs, entropy = self.get_current_log_probs(self.states_batch, self.actions_batch)
            ratios = torch.exp(curr_log_probs - self.log_probs_batch)
            loss_1 = ratios * advantages
            loss_2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            entropy_loss = - entropy.mean()
            actor_loss = (-torch.min(loss_1, loss_2)).mean() + self.ent_coeff * entropy_loss
            critic_loss = self.loss_fn(values, returns)

            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        print(f"actor: {actor_loss.detach():.6f} critic: {critic_loss.detach():.2f} entropy: {-entropy_loss:.2f}")

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass

    def update_epsilon(self):
        pass

    def reset(self):
        self.deck = np.zeros((10, 4))

    def save(self, path):
        """Saves policy network's state dictionary to path"""
        torch.save(self.actor_net.state_dict(), path)
        torch.save(self.critic_net.state_dict(), path)

    def load(self, path):
        """Loads policy network's state dictionary from path"""
        self.actor_net.load_state_dict(torch.load(path))
        self.critic_net.load_state_dict(torch.load(path))
