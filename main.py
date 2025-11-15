import gymnasium as gym
import kymnasium as kym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
import pickle
from collections import deque
from typing import Any, Dict, Tuple

# =================================================
# ⚙️ V7 하이퍼파라미터 및 경로 정의 (안정성 강화)
# =================================================
GAMMA = 0.990 # ⬆️ 장기 목표(120초)를 위해 감마 상향 (0.98 -> 0.99)
LEARNING_RATE = 1e-4
BUFFER_SIZE = 200000
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 300000
TARGET_UPDATE_FREQUENCY = 1000
N_ENEMIES_TO_CONSIDER = 3 # 유지
SAVE_PATH = "./models/avoid_blurp_agent_v7.pkl" # 파일명 변경
N_STEPS = 5

STATE_SIZE = 1 + N_ENEMIES_TO_CONSIDER * 4 + 2 + 1
ACTION_SIZE = 3

# --- ReplayBuffer, QNetwork, AvoidBlurpAgent 클래스 정의는 main.py의 V6 코드와 동일 ---
# (공간 절약을 위해 클래스 정의는 이전 답변의 V6 코드를 사용합니다.)

# --- 리플레이 메모리 클래스 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# --- QNetwork 클래스 정의 ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.01), nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.01), nn.Linear(128, action_size)
        )
        self.activation = nn.LeakyReLU(0.01)
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        q_values = v + (a - a.mean(dim=1, keepdim=True))
        return q_values

# --- AvoidBlurpAgent 클래스 정의 ---
class AvoidBlurpAgent(kym.Agent):
    def __init__(self, state_size, action_size, training=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.training = training
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.training:
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
            self.memory = ReplayBuffer(BUFFER_SIZE)
            self.steps_done = 0
            self.n_steps = N_STEPS
            self.gamma = GAMMA
            self.n_step_buffer = deque(maxlen=self.n_steps)
            self.prev_action = 0
            self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
            self.use_amp = torch.cuda.is_available()
        # [주의: load/act/save/preprocess_state 메서드도 V6과 동일하게 유지됩니다.]
    
    def _preprocess_state(self, obs: Dict) -> np.ndarray:
        if obs is None: return np.zeros(self.state_size)
        player = obs['player']
        enemies = obs['enemies']
        player_x = player[0]
        player_y = player[1]
        player_features = [player_x / 600.0]
        active_enemies = enemies[enemies[:, 1] > 0]
        sorted_enemies = sorted(
            active_enemies,
            key=lambda e: ((player_x - e[0])**2 + (player_y - e[1])**2)**0.5
        )
        top_enemies = sorted_enemies[:N_ENEMIES_TO_CONSIDER]
        enemy_features = []
        min_dist = float('inf')
        for enemy in top_enemies:
            enemy_x_norm = enemy[0] / 600.0
            enemy_y_norm = enemy[1] / 750.0
            enemy_vel_y = enemy[4]; enemy_accel_y = enemy[5]
            enemy_features.extend([enemy_x_norm, enemy_y_norm, enemy_vel_y, enemy_accel_y])
            x_dist = abs(player_x - enemy[0])
            if enemy[1] > 600: min_dist = min(min_dist, x_dist)
        num_pad = N_ENEMIES_TO_CONSIDER - len(top_enemies)
        if num_pad > 0: enemy_features.extend([0.0] * (num_pad * 4))
        prev_action_one_hot = [0.0, 0.0]
        if self.training and hasattr(self, 'prev_action'):
            if self.prev_action == 1: prev_action_one_hot = [1.0, 0.0]
            elif self.prev_action == 2: prev_action_one_hot = [0.0, 1.0]
        min_dist_norm = min(min_dist / 600.0, 1.0) if min_dist != float('inf') else 1.0
        state_vector = np.array(player_features + enemy_features + prev_action_one_hot + [min_dist_norm], dtype=np.float32)
        return state_vector

    def act(self, observation: Any, info: Dict) -> int:
        if self.training:
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
            self.steps_done += 1
            if random.random() < eps_threshold: return random.randrange(self.action_size)
        with torch.no_grad():
            state = self._preprocess_state(observation)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        with open(path, 'rb') as f: agent = pickle.load(f)
        agent.policy_net.eval()
        return agent

    def save(self, path: str):
        dir_path = os.path.dirname(path)
        if dir_path: os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f: pickle.dump(self, f)
        
    def _get_n_step_info(self, buffer_slice):
        n_step_reward = 0.0
        gamma_power = 1.0
        (s_0, a_0, _, _, _) = buffer_slice[0]
        (_, _, _, final_next_s, final_done) = buffer_slice[-1]
        for i in range(len(buffer_slice)):
            (s, a, r, next_s, d) = buffer_slice[i]
            n_step_reward += gamma_power * r
            gamma_power *= self.gamma
            if d: final_next_s = next_s; final_done = True; break
        return (s_0, a_0, n_step_reward, final_next_s, final_done)

    def push_to_memory(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps: return
        s_0, a_0, n_step_reward, final_next_s, final_done = self._get_n_step_info(self.n_step_buffer)
        self.memory.push(s_0, a_0, n_step_reward, final_next_s, final_done)

    def flush_n_step_buffer(self):
        while len(self.n_step_buffer) > 0:
            s_0, a_0, n_step_reward, final_next_s, final_done = self._get_n_step_info(self.n_step_buffer)
            self.memory.push(s_0, a_0, n_step_reward, final_next_s, final_done)
            self.n_step_buffer.popleft()
            
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE: return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        state_batch = torch.from_numpy(np.vstack(batch[0])).to(self.device)
        action_batch = torch.tensor(batch[1], device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], device=self.device, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.vstack(batch[3])).to(self.device)
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.float32).unsqueeze(1)
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                q_values = self.policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad(): next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1); next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
                gamma_n = self.gamma ** self.n_steps
                expected_q_values = reward_batch + (gamma_n * next_q_values * (1 - done_batch))
                loss = nn.SmoothL1Loss()(q_values, expected_q_values)
            self.optimizer.zero_grad(); self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); self.scaler.step(self.optimizer); self.scaler.update()
        else:
            q_values = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad(): next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1); next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            gamma_n = self.gamma ** self.n_steps
            expected_q_values = reward_batch + (gamma_n * next_q_values * (1 - done_batch))
            loss = nn.SmoothL1Loss()(q_values, expected_q_values)
            self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); self.optimizer.step()

# --- V7 보상 함수 ---
def calculate_reward(obs, done, action):
    if done: return -100.0

    reward = 1.0 # ⬆️ 생존 보상 강화 (+0.1 -> +1.0)

    # 정지 보상 제거 (기본 생존 보상이 높으므로 정지 이점은 자연스럽게 학습됨)
    # if action == 0: reward += 0.5 

    player = obs['player']
    player_x = player[0]
    player_width = player[2]
    player_y = player[1]
    enemies = obs['enemies']
    active_enemies = enemies[enemies[:, 1] > 0]

    # --- 임박 위험 페널티 ---
    imminent_danger_penalty = 0.0
    for enemy in active_enemies:
        ex, ey, e_width, _, _, _ = enemy
        if ey < player_y:
            x_dist = abs(player_x - ex)
            collision_threshold = (player_width + e_width) / 2
            if x_dist < collision_threshold + 20: 
                closeness_factor = ey / 750.0
                penalty = -3.0 * (closeness_factor ** 3) 
                imminent_danger_penalty += penalty
    
    reward += imminent_danger_penalty

    # --- 비선형 경계 페널티 복구 및 강화 ---
    # 화면 너비 600 기준 (0 ~ 600)
    
    # 50px 경계 내에 있을 경우 페널티 부여
    if player_x < 50: 
        dist_left = 50 - player_x 
        # 🚨 페널티 강화: 거리의 제곱에 비례하여 페널티 증가
        reward -= 0.1 * (dist_left ** 2) 
    
    if player_x > 550:
        dist_right = player_x - 550
        # 🚨 페널티 강화: 거리의 제곱에 비례하여 페널티 증가
        reward -= 0.1 * (dist_right ** 2)
        
    return reward

def train():
    print("=" * 60)
    print("🚀 AvoidBlurp DQN 훈련 시작 (V7: 생존/안정성 강화)")
    print("=" * 60)

    env = gym.make(id='kymnasium/AvoidBlurp-Normal-v0', render_mode='rgb_array', obs_type='custom')
    
    # 모델 로드 시도 (이어서 훈련 가능)
    if os.path.exists(SAVE_PATH):
        agent = AvoidBlurpAgent.load(SAVE_PATH)
        agent.training = True 
        print(f"훈련 재개: Steps Done={agent.steps_done}")
    else:
        agent = AvoidBlurpAgent(STATE_SIZE, ACTION_SIZE, training=True)
        print("--- 새 모델 훈련 시작 ---")

    num_episodes = 5000
    scores = []
    max_score = 0.0

    for i_episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = agent._preprocess_state(obs)
        done = False
        score = 0

        while not done:
            action = agent.act(obs, info)
            agent.prev_action = action
            
            # 주의: info 인자는 reward 계산에 필요하지 않으므로 제거
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get('time_elapsed', 0)

            # V7 보상 함수 적용
            reward = calculate_reward(obs, done, action) 

            next_state = agent._preprocess_state(next_obs)
            agent.push_to_memory(state, action, reward, next_state, done)
            state = next_state
            obs = next_obs
            agent.optimize_model()

        agent.flush_n_step_buffer()
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        max_score = max(max_score, score)

        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * agent.steps_done / EPSILON_DECAY)

        print(f"Episode {i_episode}/{num_episodes} | 생존: {score:.2f}초 (최고: {max_score:.2f}초, 평균(100): {avg_score:.2f}초) | Epsilon: {eps_threshold:.3f}")

        if agent.steps_done % TARGET_UPDATE_FREQUENCY == 0 and agent.steps_done > 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"  🎯 Target network updated! (Step {agent.steps_done})")

        if score >= 120:
            print(f"\n🎉 2분 달성! Episode {i_episode}에서 {score:.2f}초 생존!")
            break

    agent.save(SAVE_PATH)
    env.close()
    return agent


if __name__ == "__main__":
    train()