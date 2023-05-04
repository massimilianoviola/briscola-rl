import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from typing import List


class ActorCritic(nn.Module):
    """Shared layers for actor and critic"""
    def __init__(self, state_dim, action_dim, actor_layers, critic_layers):
        super().__init__()
        # Given the state outputs actions probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_layers[0]),
            nn.ReLU(),
            nn.Linear(actor_layers[0], actor_layers[1]),
            nn.ReLU(),
            nn.Linear(actor_layers[1], action_dim),
            nn.Softmax(),
        )
        # Given the state outputs its value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, critic_layers[0]),
            nn.ReLU(),
            nn.Linear(critic_layers[0], critic_layers[1]),
            nn.ReLU(),
            nn.Linear(critic_layers[1], 1),
        )

    def forward(self, x):
        actions_probabilities = self.actor(x)
        state_value = self.critic(x)
        return actions_probabilities, state_value


class PPOAgent:
    """PPO with shared networks to represent policy and value function"""
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        discount: float,
        gae_lambda: float,
        critic_loss_fn,
        learning_rate: float = 1e-3,
        actor_layers: List[int] = [256, 256],
        critic_layers: List[int] = [256, 256],
        ppo_steps: int = 5,
        ppo_clip: float = 0.2,
        ent_coeff: float = 0.01,
        value_coeff: float = 0.5,
        num_episodes: int = 64,
        device=None,
        log=True,
    ) -> None:
        self.name = "PPOAgent"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.policy = ActorCritic(
            n_features, n_actions, actor_layers, critic_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.discount = discount
        self.loss_fn = critic_loss_fn
        self.ppo_steps = ppo_steps
        self.ppo_clip = ppo_clip
        self.ent_coeff = ent_coeff
        self.value_coeff = value_coeff
        self.num_episodes = num_episodes

        self.reset_batch()

        self.last_state = None
        self.state = None
        self.action = None
        self.log_prob = None
        self.reward = None

        self.log = log
        self.ep = 0
        self.max_takes = 10

    def reset_batch(self):
        self.states_batch = []
        self.actions_batch = []
        self.log_probs_batch = []
        self.episode_rewards = []
        self.rewards_batch = []
        self.dones = []

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
        #state = np.concatenate((state, deck))

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def select_action(self, available_actions):
        state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        actions_probabilities, _ = self.policy(state)
        dist = distributions.Categorical(actions_probabilities)

        takes = 0
        while True:
            takes += 1
            if takes >= self.max_takes:
                action = np.random.choice(available_actions)
                self.action = torch.as_tensor(action)
                self.log_prob = dist.log_prob(self.action).detach()
                break
            self.action = dist.sample().detach()
            self.log_prob = dist.log_prob(self.action).detach()
            if self.action in available_actions:
                break

        return self.action.item()

    def update(self, reward):
        self.reward = reward

        self.states_batch.append(self.last_state)
        self.actions_batch.append(self.action)
        self.log_probs_batch.append(self.log_prob)
        self.episode_rewards.append(self.reward)
        self.dones.append(self.done)
        if self.done:
            self.ep += 1
            self.rewards_batch.append(self.episode_rewards)
            self.episode_rewards = []

            if len(self.rewards_batch) >= self.num_episodes:
                self.learn()
                self.reset_batch()

    def get_returns(self, rewards_batch, normalize=True):
        returns = []
        for episode_rewards in reversed(rewards_batch):
            G = 0
            for r in reversed(episode_rewards):
                G = r + self.discount * G
                returns.append(G)
        returns = returns[::-1]
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        return returns

    def get_advantages(self, returns, values, normalize=True):
        adv = returns - values.detach().squeeze()
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        return adv

    def gae_advantages(self, rewards_batch, values, dones, normalize=True):
        v = values.detach()
        rewards = rewards_batch.flatten()
        adv = np.zeros_like(rewards)
        gae_cum = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_v = 0.0
                d = 0
            else:
                next_v = v[i+1]
                d = 1
            delta = rewards[i] + self.discount * next_v * d - v[i]
            gae_cum = delta + self.discount * self.gae_lambda * gae_cum * d
            adv[i] = gae_cum

        adv = torch.Tensor(adv).to(self.device)
        ret = adv + v

        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
            ret = (ret - ret.mean()) / (ret.std() + 1e-10)

        return adv, ret

    def evaluate(self, states_batch, actions_batch):
        actions_probabilities, state_values = self.policy(states_batch)
        dist = distributions.Categorical(actions_probabilities)
        log_probs = dist.log_prob(actions_batch)
        return log_probs, state_values.squeeze(), dist.entropy()

    def learn(self):
        self.states_batch = (
            torch.tensor(np.array(self.states_batch), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.actions_batch = (
            torch.tensor(np.array(self.actions_batch), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.log_probs_batch = (
            torch.tensor(np.array(self.log_probs_batch), dtype=torch.float32)
            .detach()
            .to(self.device)
        )

        returns = self.get_returns(self.rewards_batch)
        _, values = self.policy(self.states_batch)
        adv = self.get_advantages(returns, values)
        #breakpoint()
        for _ in range(self.ppo_steps):
            curr_log_probs, values, entropy = self.evaluate(
                self.states_batch, self.actions_batch
            )

            ratios = torch.exp(curr_log_probs - self.log_probs_batch)
            loss_1 = ratios * adv
            loss_2 = torch.clamp(ratios, 1-self.ppo_clip, 1+self.ppo_clip)*adv

            policy_loss = -torch.min(loss_1, loss_2).mean()
            value_loss = self.loss_fn(returns, values)
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.value_coeff * value_loss
                + self.ent_coeff * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.log:
            print(
                f"EPISODE #{self.ep} policy_loss: {policy_loss:.4f} value_loss: {value_loss:.4f} entropy_loss: {entropy_loss:.2f}",
                end='\r'
            )

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
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """Loads policy network's state dictionary from path"""
        self.policy.load_state_dict(torch.load(path))
