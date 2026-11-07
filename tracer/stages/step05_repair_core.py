"""Unchanged active CH-GAIL classes and repair recurrence from the five-run source."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
DT = 0.1
GAIL_EPOCHS = 60

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, x):
        return self.net(x)

    def act(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            a = self.forward(s).cpu().numpy()[0]
        return a

class Discriminator(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

class GAILPair:

    def __init__(self, state_dim=3, action_dim=1, device='cpu', lr_policy=0.001, lr_disc=0.001, use_state_noise=False, noise_std=0.1):
        self.device = device
        self.use_state_noise = use_state_noise
        self.noise_std = noise_std
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.disc = Discriminator(state_dim, action_dim).to(device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=lr_disc)

    def _maybe_add_noise(self, s):
        if not self.use_state_noise:
            return s
        return s + torch.randn_like(s) * self.noise_std

    def train_pair(self, expert_states, expert_actions, epochs=60, batch=64):
        dataset = np.concatenate([expert_states, expert_actions], axis=1)
        dataset = torch.tensor(dataset, dtype=torch.float32).to(self.device)
        for ep in range(epochs):
            idx = torch.randperm(len(dataset))
            dataset_shuf = dataset[idx]
            for start in range(0, len(dataset_shuf), batch):
                batch_data = dataset_shuf[start:start + batch]
                s_exp = batch_data[:, :3]
                a_exp = batch_data[:, 3:].reshape(-1, 1)
                a_gen = self.policy(s_exp)
                s_disc = self._maybe_add_noise(s_exp)
                d_real = self.disc(s_disc, a_exp)
                d_fake = self.disc(s_disc, a_gen.detach())
                disc_loss = -torch.mean(torch.log(d_real + 1e-08) + torch.log(1 - d_fake + 1e-08))
                self.disc_opt.zero_grad()
                disc_loss.backward()
                self.disc_opt.step()
                d_fake2 = self.disc(s_exp, a_gen)
                policy_loss = -torch.mean(torch.log(d_fake2 + 1e-08))
                self.policy_opt.zero_grad()
                policy_loss.backward()
                self.policy_opt.step()

def repair_trajectory(hgail, abnormal_states, leader_speeds):
    repaired = []
    repaired_actions = []
    current_s, current_v, current_rel_v = abnormal_states[0]
    for t in range(len(abnormal_states)):
        a = hgail.act([current_s, current_v, current_rel_v])
        v_next = max(a[0], 0.0)
        v_leader = leader_speeds[t] if t < len(leader_speeds) else leader_speeds[-1]
        next_rel_v = v_leader - v_next
        next_s = max(current_s + next_rel_v * DT, 5.0)
        repaired.append([next_s, v_next, next_rel_v])
        repaired_actions.append([v_next])
        current_s, current_v, current_rel_v = (next_s, v_next, next_rel_v)
    return (np.array(repaired), np.array(repaired_actions))

class HGAIL:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = PolicyNetwork(3, 1).to(self.device)

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy(s).cpu().numpy()[0]

    def train_gailpair(self, expert_states, expert_actions):
        pair1 = GAILPair(device=self.device)
        pair2 = GAILPair(device=self.device, use_state_noise=True)
        pair1.train_pair(expert_states, expert_actions, epochs=GAIL_EPOCHS)
        pair2.train_pair(expert_states, expert_actions, epochs=GAIL_EPOCHS)
        self.policy = pair2.policy
