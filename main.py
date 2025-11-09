import gymnasium as gym
import kymnasium as kym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Any, Dict

# --- 하이퍼파라미터 및 설정 ---
TIMESTEPS = 2000000 
MODEL_DIR = "./ppo_ultimate_model/" 
SAVE_PATH = os.path.join(MODEL_DIR, "best_model.zip")
# ------------------------------

# ----------------------------------------------------
# 1. kymnasium.Agent 상속 구현 
# ----------------------------------------------------
class PPOAgent(kym.Agent):
    """
    PPO 에이전트를 위한 kymnasium Agent 래퍼입니다.
    """
    def __init__(self, model):
        self.model = model
        self.action_space = gym.spaces.Discrete(3)

    def act(self, observation: Any, info: Dict) -> int:
        obs_array = self._process_obs(observation)
        action, _ = self.model.predict(obs_array, deterministic=True)
        return int(action.item()) 

    def _process_obs(self, observation: Dict) -> np.ndarray:
        """관측치 딕셔너리를 단일 벡터로 변환합니다."""
        player_vec = observation["player"].flatten()
        enemies_vec = observation["enemies"].flatten()
        return np.concatenate([player_vec, enemies_vec], dtype=np.float32)

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        env_for_load = DummyVecEnv([make_env])
        # Learning Rate 0.0003으로 로드 시도
        loaded_model = PPO.load(path, env=env_for_load, learning_rate=0.0003)
        return cls(loaded_model)

    def save(self, path: str):
        self.model.save(path)

# ----------------------------------------------------
# 2. 커스텀 환경 래퍼 (Boundary Penalty 상향)
# ----------------------------------------------------
class AvoidBlurpRewardWrapper(gym.RewardWrapper):
    """
    Reward Shaping을 통해 벽 비비기를 방지하고 움직임을 복구합니다.
    """
    def __init__(self, env):
        super().__init__(env)
        self.LIVE_REWARD = 0.5    # 매 스텝 생존 보상
        self.DEATH_PENALTY = -100.0
        
        # ✅ FINAL ADJUSTMENT: Boundary Penalty Factor를 4배 상향
        self.BOUNDARY_PENALTY_FACTOR = 0.2
        self.LEFT_BOUNDARY = 0
        self.RIGHT_BOUNDARY = 750

    def reward(self, rew):
        new_reward = self.LIVE_REWARD
        
        try:
            raw_obs = self.env.unwrapped.last_raw_obs
            player_x = raw_obs["player"][0]
        except AttributeError:
            return new_reward

        # --- A. 벽 비비기/경계 페널티 ---
        # 50px 경계 내에 있을 경우 페널티 부여
        boundary_distance_left = player_x - self.LEFT_BOUNDARY
        boundary_distance_right = self.RIGHT_BOUNDARY - player_x
        
        if boundary_distance_left < 50: 
            # 왼쪽 경계에 가까울수록 더 큰 페널티 (0.2 * 거리)
            new_reward -= self.BOUNDARY_PENALTY_FACTOR * (50 - boundary_distance_left)
        
        if boundary_distance_right < 50: 
            # 오른쪽 경계에 가까울수록 더 큰 페널티
            new_reward -= self.BOUNDARY_PENALTY_FACTOR * (50 - boundary_distance_right)
            
        return new_reward

# ----------------------------------------------------
# 3. 환경 관측치 변환 래퍼 (데이터 저장)
# ----------------------------------------------------
class AvoidBlurpObsWrapper(gym.ObservationWrapper):
    """원본 딕셔너리를 환경에 저장하고, SB3 학습을 위한 벡터 변환을 수행합니다."""
    def __init__(self, env):
        super().__init__(env)
        obs_dim = 5 + (30 * 6)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def observation(self, obs):
        self.env.unwrapped.last_raw_obs = obs 
        player_vec = obs["player"].flatten()
        enemies_vec = obs["enemies"].flatten()
        return np.concatenate([player_vec, enemies_vec], dtype=np.float32)

# ----------------------------------------------------
# 4. Action 저장 래퍼 
# ----------------------------------------------------
class AvoidBlurpActionWrapper(gym.ActionWrapper):
    """직전에 선택된 Action을 환경에 저장합니다."""
    def __init__(self, env):
        super().__init__(env)
        self.env.unwrapped.last_action = 0 

    def action(self, act):
        self.env.unwrapped.last_action = act
        return act

# ----------------------------------------------------
# 5. 환경 생성 및 훈련 함수
# ----------------------------------------------------
def make_env():
    """환경 인스턴스를 생성하고 래퍼를 적용합니다."""
    env_id = 'kymnasium/AvoidBlurp-Normal-v0'
    env = gym.make(
        id=env_id,
        render_mode='rgb_array', 
        bgm=False,
        obs_type='custom'
    )
    env = AvoidBlurpActionWrapper(env)
    env = AvoidBlurpObsWrapper(env)
    env = AvoidBlurpRewardWrapper(env)
    return env

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    def make_env_wrapper():
        return make_env()

    # 모델 로드 시도 및 Critic 복구 로직
    try:
        env = DummyVecEnv([make_env_wrapper for _ in range(4)])
        # 기존 모델 로드 시도 (Learning Rate 0.0003으로 강제)
        model = PPO.load(SAVE_PATH, env=env, learning_rate=0.0003)
        print(f"--- 기존 모델 로드 성공! ({SAVE_PATH}) Critic 복구 훈련을 시작합니다. ---")
    except Exception:
        # 모델 로드 실패 시 새로 시작 
        env = DummyVecEnv([make_env_wrapper for _ in range(4)])
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003,      
            gamma=0.999,
            n_steps=1024,              
            batch_size=256,            
            n_epochs=10,               
            ent_coef=0.05,             # ✅ Entropy 상향: 구석에 갇히지 않도록 탐험 강제
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            device="auto"
        )
        print("--- 모델 로드 실패 또는 파일 없음. 200만 스텝 새로운 훈련을 시작합니다. ---")


    # 훈련 중 일정 간격으로 모델 저장
    checkpoint_callback = CheckpointCallback(
        save_freq=40960, 
        save_path=MODEL_DIR,
        name_prefix="ppo_ultimate_checkpoint"
    )
    
    # 훈련 목표 설정
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)

    # 최종 모델 저장
    model.save(SAVE_PATH)
    print(f"--- 훈련 완료. 모델이 다음 위치에 저장됨: {SAVE_PATH} ---")
    
    return SAVE_PATH

if __name__ == '__main__':
    train()
