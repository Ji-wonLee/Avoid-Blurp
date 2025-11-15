# test.py (최종 버전)

import kymnasium as kym
import os
import pickle
import torch
import numpy as np
from typing import Any, Dict

# 🚨 해결책: main.py에서 필요한 클래스와 변수를 명시적으로 import 합니다.
# __main__ 문제가 발생하지 않도록 필요한 모든 클래스를 불러옵니다.
from main import (
    AvoidBlurpAgent, SAVE_PATH, 
    QNetwork, ReplayBuffer, calculate_reward, 
    N_ENEMIES_TO_CONSIDER, STATE_SIZE 
)

if __name__ == '__main__':
    
    print("\n" + "=" * 60)
    print("🏁 V7 에이전트 시연 준비 (탐험 비활성화)")
    print("=" * 60)

    # 1. 에이전트 로드 (가장 간결한 형태)
    try:
        # AvoidBlurpAgent와 SAVE_PATH 변수를 main.py에서 가져옴
        agent = AvoidBlurpAgent.load(SAVE_PATH) 
    except FileNotFoundError:
        print(f"🚨 모델 로드 실패: {SAVE_PATH} 파일을 찾을 수 없습니다. main.py를 먼저 실행하여 훈련하세요.")
        exit()
    except Exception as e:
        # QNetwork, ReplayBuffer 등의 클래스 누락으로 인한 로드 실패 오류를 잡아냅니다.
        print(f"❌ 에이전트 로드 중 오류 발생: {e}")
        exit()

    print("\n[평가 시작 - 시연 화면을 확인하세요]")
    
    # 2. 평가 실행 (요청하신 구문)
    kym.evaluate( 
        env_id='kymnasium/AvoidBlurp-Normal-v0', 
        agent=agent, 
        render_mode='human', 
        bgm=True, 
        obs_type='custom',
        # num_episodes 매개변수 생략 (기본값 1회 시도로 실행)
    )

    print("\n평가 1회 시도가 완료되었습니다. (다음 시도를 위해 명령을 다시 실행해주세요)")
    print("=" * 60)