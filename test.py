import kymnasium as kym
# PPOAgent 클래스가 정의된 실제 파일명(main.py)으로 import 합니다.
from main import PPOAgent 
import os

# 훈련된 모델이 저장된 폴더 경로를 'ppo_optimized_model'로 설정합니다.
MODEL_DIR = "./ppo_ultimate_model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model")

# ----------------------------------------------------
# RL Competition 평가 코드 (교수님 지침 반영)
# ----------------------------------------------------
def evaluate_agent():
    import kymnasium as kym # kymnasium import는 유지

    print("--- 에이전트 로드 및 시연 시작 ---")

    # 1. 에이전트 불러오기
    # 모델 로드 시도
    agent = PPOAgent.load(MODEL_PATH)
    
    # 2. kym.evaluate() 함수를 사용하여 에이전트 시연
    kym.evaluate(
        env_id='kymnasium/AvoidBlurp-Normal-v0',
        agent=agent,
        render_mode='human',
        bgm=True,
        obs_type='custom'
    )
    print("--- 시연 완료: 결과는 화면에 출력됩니다 ---")

if __name__ == '__main__':
    evaluate_agent()
