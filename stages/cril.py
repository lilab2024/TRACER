import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
STATE_COLS = ['Spacing', 'Speed_Follower', 'Relative_speed']
ACTION_COL = 'Speed_Follower'
DT = 0.1
OUTPUT_PATH = 'repaired_output.txt'
LEADER_FILE = 'normal_leader_0.txt'
FAKE_MIDLEVEL = True
BC_PRETRAIN = True
GEN_EPOCHS = 60
BC_EPOCHS = 80
GAIL_EPOCHS = 60
WINDOW_SIZE = 8
BATCH_SIZE = 64
(BETA1_MIN, BETA1_MAX) = (0.061533, 0.078499)
(BETA3_MIN, BETA3_MAX) = (-0.075789, -0.058077)

class GeneratorNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        (out, _) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

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

    def __init__(self, state_dim=3, action_dim=1, device='cpu', lr_policy=0.001, lr_disc=0.001, use_state_noise=False, noise_std=0.1, state_mean=None, state_std=None):
        self.device = device
        self.use_state_noise = use_state_noise
        self.noise_std = noise_std
        self.state_mean = None if state_mean is None else torch.tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std = None if state_std is None else torch.tensor(state_std, dtype=torch.float32, device=device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.disc = Discriminator(state_dim, action_dim).to(device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=lr_disc)

    def _maybe_add_noise(self, s):
        if not self.use_state_noise:
            return s
        if self.state_mean is None or self.state_std is None:
            noise = torch.randn_like(s) * self.noise_std
            return s + noise
        safe_std = torch.clamp(self.state_std, min=1e-08)
        s_normalized = (s - self.state_mean) / safe_std
        noise = torch.randn_like(s_normalized) * self.noise_std
        s_noisy_normalized = s_normalized + noise
        return s_noisy_normalized * safe_std + self.state_mean

    def train_pair(self, expert_states, expert_actions, epochs=60, batch=64, verbose=True):
        dataset = np.concatenate([expert_states, expert_actions], axis=1)
        dataset = torch.tensor(dataset, dtype=torch.float32).to(self.device)
        for ep in range(epochs):
            idx = torch.randperm(len(dataset))
            dataset_shuf = dataset[idx]
            for start in range(0, len(dataset_shuf), batch):
                batch_data = dataset_shuf[start:start + batch]
                s_exp = batch_data[:, :3]
                a_exp = batch_data[:, 3:].reshape(-1, 1)
                s_disc = self._maybe_add_noise(s_exp)
                a_gen = self.policy(s_disc)
                d_real = self.disc(s_disc, a_exp)
                d_fake = self.disc(s_disc, a_gen.detach())
                disc_loss = -torch.mean(torch.log(d_real + 1e-08) + torch.log(1 - d_fake + 1e-08))
                self.disc_opt.zero_grad()
                disc_loss.backward()
                self.disc_opt.step()
                a_gen2 = self.policy(s_disc)
                d_fake2 = self.disc(s_disc, a_gen2)
                policy_loss = -torch.mean(torch.log(d_fake2 + 1e-08))
                self.policy_opt.zero_grad()
                policy_loss.backward()
                self.policy_opt.step()
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f'[Pair Train | noise={self.use_state_noise}] Epoch {ep + 1}/{epochs} DiscLoss={disc_loss.item():.6f} PolicyLoss={policy_loss.item():.6f}')
        print('=== Pair training finished ===')

def fine_tune_G2_with_D1(policy2, disc1, expert_states, epochs=60, batch=64, lr=0.0001, device='cpu'):
    policy2.to(device)
    disc1.to(device)
    opt = optim.Adam(policy2.parameters(), lr=lr)
    states = torch.tensor(expert_states, dtype=torch.float32).to(device)
    for ep in range(epochs):
        idx = torch.randperm(len(states))
        states = states[idx]
        for start in range(0, len(states), batch):
            s = states[start:start + batch]
            a = policy2(s)
            d = disc1(s, a)
            loss = -torch.mean(torch.log(d + 1e-08))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'[Stage C] Epoch {ep + 1}/{epochs} Loss={loss.item():.6f}')

class HGAIL:

    def __init__(self, state_dim=3, action_dim=1, device=None):
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.disc = Discriminator(state_dim, action_dim).to(self.device)
        self.generator = GeneratorNetwork(input_dim=state_dim, output_dim=state_dim).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=0.001)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=0.001)
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=0.001)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def build_windows(self, states, window_size=WINDOW_SIZE):
        (N, D) = states.shape
        if N < window_size:
            return (np.zeros((0, window_size, D), dtype=np.float32), np.zeros((0, D), dtype=np.float32))
        (X, Y) = ([], [])
        for i in range(window_size - 1, N):
            window = states[i - window_size + 1:i + 1]
            X.append(window.copy())
            Y.append(states[i].copy())
        return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32))

    def train_generator(self, expert_states, epochs=GEN_EPOCHS, batch_size=BATCH_SIZE):
        (X, Y) = self.build_windows(expert_states, WINDOW_SIZE)
        if len(X) == 0:
            print('=== Generator training skipped: not enough expert data ===')
            return
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        dataset_size = len(X)
        print(f'=== Training Generator on {dataset_size} windows ===')
        for ep in range(epochs):
            perm = torch.randperm(dataset_size)
            X = X[perm]
            Y = Y[perm]
            for start in range(0, dataset_size, batch_size):
                xb = X[start:start + batch_size]
                yb = Y[start:start + batch_size]
                pred = self.generator(xb)
                loss = self.mse_loss(pred, yb)
                self.gen_opt.zero_grad()
                loss.backward()
                self.gen_opt.step()
            if ep % max(1, epochs // 5) == 0:
                print(f'[Gen Epoch {ep}/{epochs}] MSE={loss.item():.6f}')
        print('=== Generator training finished ===')

    def bc_pretrain(self, expert_states, expert_actions, epochs=BC_EPOCHS, batch_size=BATCH_SIZE, use_gen=FAKE_MIDLEVEL):
        if use_gen:
            (X, Y_states) = self.build_windows(expert_states, WINDOW_SIZE)
            if len(X) == 0:
                print('=== BC skipped: not enough windows ===')
                return
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                inputs = self.generator(X_t).cpu().numpy()
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            actions = torch.tensor(expert_actions[WINDOW_SIZE - 1:], dtype=torch.float32).to(self.device)
        else:
            inputs = torch.tensor(expert_states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(expert_actions, dtype=torch.float32).to(self.device)
        dataset_size = inputs.shape[0]
        print(f'=== BC pretrain on {dataset_size} samples (use_gen={use_gen}) ===')
        for ep in range(epochs):
            perm = torch.randperm(dataset_size)
            inputs = inputs[perm]
            actions = actions[perm]
            for start in range(0, dataset_size, batch_size):
                s_batch = inputs[start:start + batch_size]
                a_batch = actions[start:start + batch_size]
                pred = self.policy(s_batch)
                loss = self.mse_loss(pred, a_batch)
                self.policy_opt.zero_grad()
                loss.backward()
                self.policy_opt.step()
            if ep % max(1, epochs // 5) == 0:
                print(f'[BC Epoch {ep}/{epochs}] BC_loss={loss.item():.6f}')
        print('=== BC pretraining finished ===')

    def train_gailpair(self, expert_states, expert_actions, use_gen=FAKE_MIDLEVEL):
        state_mean = np.mean(expert_states, axis=0).astype(np.float32)
        state_std = np.std(expert_states, axis=0).astype(np.float32)
        pair1 = GAILPair(state_dim=3, action_dim=1, device=self.device, use_state_noise=False, state_mean=state_mean, state_std=state_std)
        pair2 = GAILPair(state_dim=3, action_dim=1, device=self.device, use_state_noise=True, state_mean=state_mean, state_std=state_std)
        pair1.train_pair(expert_states, expert_actions, epochs=GAIL_EPOCHS, batch=BATCH_SIZE)
        pair2.train_pair(expert_states, expert_actions, epochs=GAIL_EPOCHS, batch=BATCH_SIZE)
        fine_tune_G2_with_D1(pair2.policy, pair1.disc, expert_states, epochs=GAIL_EPOCHS, batch=BATCH_SIZE, device=self.device)
        self.policy = pair2.policy
        print('=== GAILPair training + Stage C finished ===')

    def act(self, state, use_gen_for_act=FAKE_MIDLEVEL):
        s_np = np.array(state, dtype=np.float32)
        if use_gen_for_act:
            w = np.tile(s_np, (WINDOW_SIZE, 1))[np.newaxis, ...]
            w_t = torch.tensor(w, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                s_clean = self.generator(w_t).cpu()
            with torch.no_grad():
                a = self.policy(s_clean.to(self.device)).cpu().numpy()[0]
        else:
            with torch.no_grad():
                a = self.policy(torch.tensor(s_np, dtype=torch.float32).unsqueeze(0).to(self.device)).cpu().numpy()[0]
        return a

def compute_beta_parameters(states):
    spacing = states[:, 0]
    speed = states[:, 1]
    rel_speed = states[:, 2]
    (v0, T, s0, delta, alpha, beta_val) = (39.12, 0.96, 10.0, 20.0, 1.18, 8.9)
    (beta1_list, beta3_list, new_flag1, new_flag3) = ([], [], [], [])
    for (s, v, dv) in zip(spacing, speed, rel_speed):
        s_star = s0 + v * T + v * dv / (2 * np.sqrt(alpha * beta_val))
        beta1 = 2 * alpha * s_star ** 2 / s ** 3
        beta3 = -alpha * (delta * v ** (delta - 1) / v0 ** delta + 2 * s_star / s ** 2 * (T + dv / (2 * np.sqrt(alpha * beta_val))))
        beta1_list.append(beta1)
        beta3_list.append(beta3)
        new_flag1.append('正常' if BETA1_MIN <= beta1 <= BETA1_MAX else '异常')
        new_flag3.append('正常' if BETA3_MIN <= beta3 <= BETA3_MAX else '异常')
    return (np.array(beta1_list), np.array(beta3_list), np.array(new_flag1), np.array(new_flag3))

def repair_trajectory(hgail, abnormal_states, leader_speeds, use_gen_for_act=FAKE_MIDLEVEL):
    repaired = []
    repaired_actions = []
    (current_s, current_v, current_rel_v) = abnormal_states[0]
    for t in range(len(abnormal_states)):
        a = hgail.act([current_s, current_v, current_rel_v], use_gen_for_act)
        v_next = max(a[0], 0.0)
        v_leader = leader_speeds[t] if t < len(leader_speeds) else leader_speeds[-1]
        next_rel_v = v_leader - v_next
        next_s = max(current_s + next_rel_v * DT, 5.0)
        repaired.append([next_s, v_next, next_rel_v])
        repaired_actions.append([v_next])
        (current_s, current_v, current_rel_v) = (next_s, v_next, next_rel_v)
    return (np.array(repaired), np.array(repaired_actions))

def save_results(original_states, repaired_states, repaired_actions, beta1, beta3, new_flag1, new_flag3, original_df, leader_speeds):
    leader_speed_matched = [leader_speeds[i] if i < len(leader_speeds) else leader_speeds[-1] for i in range(len(original_states))]
    df_out = pd.DataFrame({'Time': original_df['Time'].values, 'Leader_Speed': leader_speed_matched, 'Original_Spacing': original_states[:, 0], 'Original_Speed': original_states[:, 1], 'Original_RelSpeed': original_states[:, 2], 'Original_flag1': original_df['flag1'].values, 'Original_flag2': original_df['flag2'].values, 'Repaired_Spacing': repaired_states[:, 0], 'Repaired_Speed': repaired_states[:, 1], 'Repaired_RelSpeed': repaired_states[:, 2], 'Repaired_Action_Speed': repaired_actions[:, 0], 'new_flag1': new_flag1, 'new_flag3': new_flag3, 'beta1': beta1, 'beta3': beta3})
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(df_out.to_string(index=False, justify='right', float_format='{:10.4f}'.format))
    print(f'=== 修复数据已保存到：{OUTPUT_PATH} ===')

def main():
    print('=== 离线 H-GAIL 轨迹修复系统（动作=速度，输出新beta_flag） ===')
    df = pd.read_csv('attack_beta_flag.txt', sep='\\s+')
    expert_df = df[(df['flag1'] == '正常') & (df['flag2'] == '正常')]
    expert_states = expert_df[STATE_COLS].values.astype(np.float32)
    expert_actions = expert_df[ACTION_COL].values.astype(np.float32).reshape(-1, 1)
    abnormal_df = df[(df['flag1'] != '正常') | (df['flag2'] != '正常')]
    abnormal_states = abnormal_df[STATE_COLS].values.astype(np.float32)
    leader_data = pd.read_csv(LEADER_FILE, sep='\\s+')
    leader_speeds = leader_data.iloc[:, 1].values
    hgail = HGAIL(state_dim=3, action_dim=1)
    if FAKE_MIDLEVEL:
        hgail.train_generator(expert_states, epochs=GEN_EPOCHS)
    if BC_PRETRAIN:
        hgail.bc_pretrain(expert_states, expert_actions, epochs=BC_EPOCHS, use_gen=FAKE_MIDLEVEL)
    hgail.train_gailpair(expert_states, expert_actions, use_gen=FAKE_MIDLEVEL)
    (repaired_states, repaired_actions) = repair_trajectory(hgail, abnormal_states, leader_speeds, use_gen_for_act=FAKE_MIDLEVEL)
    (beta1, beta3, new_flag1, new_flag3) = compute_beta_parameters(repaired_states)
    save_results(abnormal_states, repaired_states, repaired_actions, beta1, beta3, new_flag1, new_flag3, abnormal_df, leader_speeds)
if __name__ == '__main__':
    main()
