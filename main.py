import gymnasium as gym
import kymnasium as kym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from typing import Any, Dict

# ----------------------------------------------------
# 1. kymnasium.Agent 상속 구현 
# ----------------------------------------------------
class PPOAgent(kym.Agent):
    """PPO 에이전트를 위한 kymnasium Agent 래퍼."""
    def __init__(self, model):
        self.model = model
        self.action_space = gym.spaces.Discrete(3)

    def act(self, observation: Any, info: Dict) -> int:
        obs_array = self._process_obs(observation)
        # model.predict는 배치 형태로 처리하므로, 배치 차원(차원 0)이 필요합니다.
        action, _ = self.model.predict(obs_array, deterministic=True)
        # IndexError 방지를 위해 .item()을 사용하여 스칼라 값을 안전하게 추출합니다.
        return int(action.item())

    def _process_obs(self, observation: Dict) -> np.ndarray:
        """
        'custom' 딕셔너리 관측치를 SB3가 처리할 수 있는 단일 벡터로 변환합니다.
        act() 및 load()에서 사용됩니다.
        """
        player_vec = observation["player"].flatten()
        enemies_vec = observation["enemies"].flatten()
        # [NOTE] player_vec (5,) + enemies_vec (180,) = (185,)
        return np.concatenate([player_vec, enemies_vec], dtype=np.float32)

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        loaded_model = PPO.load(path)
        return cls(loaded_model)

    def save(self, path: str):
        self.model.save(path)

# ----------------------------------------------------
# 2. 커스텀 환경 래퍼 및 보상 설계 (Reward Shaping 강화)
# ----------------------------------------------------
class AvoidBlurpRewardWrapper(gym.RewardWrapper):
    """
    Reward Space가 0이므로, 생존 보상과 충돌 페널티, 회피 보상을 추가합니다.
    보글보글과의 거리가 가까울수록 페널티를 급증시켜 회피를 유도합니다.
    """
    def __init__(self, env):
        super().__init__(env)
        self.LIVE_REWARD = 0.5  # 생존 보상 대폭 증가
        self.HIGH_PENALTY_FACTOR = 5000.0 # 페널티 강화 계수

    def reward(self, rew):
        new_reward = self.LIVE_REWARD

        # ObsWrapper가 저장한 원본 딕셔너리를 직접 읽어옵니다.
        if not hasattr(self.env.unwrapped, 'last_raw_obs'):
            return new_reward
            
        obs_original = self.env.unwrapped.last_raw_obs
        
        # 보상 미세 조정 (Reward Shaping): 
        # 마리오와 모든 보글보글 사이의 거리를 계산하여 가까울수록 페널티
        player_x = obs_original["player"][0]
        player_y = obs_original["player"][1]
        enemies = obs_original["enemies"]
        
        min_dist_sq = float('inf')
        
        for enemy in enemies:
            enemy_x, enemy_y = enemy[0], enemy[1]
            
            # 보글보글이 화면에 표시된 경우에만 거리를 계산
            if enemy_x != 0.0 or enemy_y != 0.0:
                # 유클리드 거리 제곱
                dist_sq = (player_x - enemy_x)**2 + (player_y - enemy_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

        if min_dist_sq != float('inf') and min_dist_sq > 1e-6:
             # 거리 제곱이 작을수록 (가까울수록) 페널티가 급증합니다.
             # 페널티가 너무 커지지 않도록 min_dist_sq에 상한을 두거나, 
             # 매우 가까울 때만 페널티를 적용하는 것이 안전합니다.
             new_reward -= (self.HIGH_PENALTY_FACTOR / min_dist_sq) 

        return new_reward

# ----------------------------------------------------
# 3. 환경 관측치 변환 래퍼 (데이터 저장 로직 추가)
# ----------------------------------------------------
class AvoidBlurpObsWrapper(gym.ObservationWrapper):
    """
    원본 딕셔너리를 환경 인스턴스에 저장하여 RewardWrapper가 사용할 수 있도록 합니다.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_dim = 5 + (30 * 6) # player (5) + enemies (180) = 185
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def observation(self, obs):
        # 원본 딕셔너리(obs)를 저장하여 RewardWrapper가 접근할 수 있도록 합니다.
        self.env.unwrapped.last_raw_obs = obs 
        
        player_vec = obs["player"].flatten()
        enemies_vec = obs["enemies"].flatten()
        return np.concatenate([player_vec, enemies_vec], dtype=np.float32)

# ----------------------------------------------------
# 4. 훈련 함수 (최종 하이퍼파라미터 조정)
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
    env = AvoidBlurpObsWrapper(env)
    env = AvoidBlurpRewardWrapper(env)
    return env

def train():
    # 시드 고정 및 환경 초기화
    SEED = 42
    set_random_seed(SEED)
    
    # 저장 경로 설정
    MODEL_DIR = "./ppo_ultimate_model/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(MODEL_DIR, "best_model.zip")

    # 벡터화 환경 생성 시 환경 복사본을 4개 사용 (데이터 수집 속도 증가)
    env = DummyVecEnv([lambda: make_env() for _ in range(4)]) 
    
    print("--- 환경 생성 완료: ULTIMATE PPO 모델 훈련 시작 ---")

    # PPO Network 아키텍처 및 하이퍼파라미터 정의
    net_arch = [dict(pi=[256, 256], vf=[256, 256])] 
    
    # [최종 수정] 과잉 행동 방지를 위한 하이퍼파라미터 조정
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.00005,    # ✅ 학습률 대폭 감소 (섬세한 조정)
        gamma=0.999,             
        n_steps=2048,            
        batch_size=512,          # ✅ 배치 사이즈 증가 (안정성 증가)
        n_epochs=10,             
        ent_coef=0.01,           
        policy_kwargs=dict(net_arch=net_arch), 
        device="auto"
    )

    # 훈련 중 일정 간격으로 모델 저장 (Checkpoint)
    checkpoint_callback = CheckpointCallback(
        save_freq=40960, 
        save_path=MODEL_DIR,
        name_prefix="ppo_ultimate_checkpoint"
    )
    
    # 총 스텝 수를 150만으로 늘려 훈련
    TIMESTEPS = 1500000 
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

    # 최종 모델 저장
    model.save(SAVE_PATH)
    print(f"--- 훈련 완료. 모델이 다음 위치에 저장됨: {SAVE_PATH} ---")
    
    return SAVE_PATH

if __name__ == '__main__':
    train()
