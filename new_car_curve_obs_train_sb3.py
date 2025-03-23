import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.gail import ExpertDataset
# from stable_baselines3.common.monitor import load_results
from stable_baselines3.sac.policies import MlpPolicy as sacmlp
from stable_baselines3.ppo.policies import MlpPolicy as ppomlp
from stable_baselines3.td3.policies import MlpPolicy as td3mlp
from stable_baselines3.ddpg.policies import MlpPolicy as ddpgmlp
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils_sb3 import SaveOnBestTrainingRewardCallback
from new_car_env_dy_obs_sb3 import CarEnv
import matplotlib
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
import torch
import time

######################
torch.autograd.set_detect_anomaly(True)
np.seterr(all='raise')
######################

goal_selection_strategy = "future"

print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("##################Running on the GPU")
else:
    device = torch.device("cpu")
    print("##################Running on the CPU")

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']



if __name__=="__main__":
    train_num = 5e7  #
    # run_mode = 'test'
    run_mode = 'train'

    method = "SAC"

    class CustomSAC(SAC):
        def train(self, gradient_steps, batch_size=128):
            super(CustomSAC, self).train(gradient_steps, batch_size)
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-2, 2)


    if run_mode == 'train':
        car_env = CarEnv()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if method == "SAC":
            log_dir = f'./SAC/new_model/{timestamp}'
            os.makedirs(log_dir, exist_ok=True)
            SAC_env = Monitor(car_env, log_dir, info_keywords=('d',))

            print('RL Train Begin...')
            Begin_time = time.time()

            policy_kwargs = dict(net_arch=dict(pi=[256, 256,  128], qf=[256, 256,  128]))
            SAC_model = SAC(policy=sacmlp, env=SAC_env, verbose=1, gamma=0.99, learning_rate=1e-4, buffer_size=int(1e5),
                            learning_starts=64, train_freq=1, batch_size=1024, tau=0.005, ent_coef='auto',
                            target_update_interval=1, device='cuda:0', policy_kwargs=policy_kwargs, use_sde=True,
                            sde_sample_freq=8)

            SAC_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

            SAC_model.learn(total_timesteps=int(train_num), callback=SAC_callback)
            print('RL Train-Finish...')
            Finish_time = time.time()
            print("the total Training time is :", (Finish_time - Begin_time) / 60)
            # print('sim.time:', sim.time)


        if method == "PPO":
            PPO_log_dir = './PPO/new_mine/'
            PPO_env = Monitor(car_env, PPO_log_dir, info_keywords=('d',))
            print('RL Train Begin...')
            policy_kwargs = dict(net_arch=dict(pi=[256, 128], qf=[256, 128]))
            PPO_Model = PPO(ppomlp, env=PPO_env, gamma=0.01, learning_rate=5e-3, n_steps=32, ent_coef=0.01, vf_coef=0.5,
                         max_grad_norm=0.5, batch_size=16, verbose=1, tensorboard_log=None, _init_setup_model=True, policy_kwargs=policy_kwargs)

            PPO_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=PPO_log_dir)
            PPO_Model.learn(total_timesteps=train_num, callback=PPO_callback)
            print('RL Train-Finish...')
            Finish_time = time.time()
            print("the total Training time is :", (Finish_time - Begin_time) / 60)

        if method=="TD3":
            TD3_log_dir = './TD3/new_mine/'
            TD3_env = Monitor(car_env, TD3_log_dir, info_keywords=('d',))
            print('RL Train Begin...')
            Begin_time = time.time()

            policy_kwargs = dict(net_arch=dict(pi=[256, 128], qf=[256, 128]))
            TD3_Model = TD3(policy=td3mlp, env=TD3_env, gamma=0.1, learning_rate=9e-4,policy_kwargs=policy_kwargs)
            TD3_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=TD3_log_dir)

            
            TD3_Model.learn(total_timesteps=int(train_num), callback=TD3_callback)
            print('RL Train-Finish...')
            Finish_time = time.time()
            print("the total Training time is :", (Finish_time - Begin_time) / 60)
            # print('sim.time:', sim.time)

    elif run_mode == 'test':
        if method=="SAC":
            vec_env = CarEnv()
            model = SAC.load(r'.\SAC\new_model\ET-MOPO_model.zip', env=vec_env)

            for ep in range(100):
                Begin_time = time.time()
                sum_r = 0
                state_list = []
                a_list = []
                a_real_list = []
                xita_list = []
                w_real_list = []
                ep_r = []
                s, info = vec_env.reset()
                t = 1
                while True:
                    vec_env.render()
                    action = model.predict(s)[0]
                    a_list.append(action[0])
                    xita_list.append(action[1])
                    s, r, done, done1, info = vec_env.step(action)
                    ep_r.append(r)
                    sum_r += r
                    Finish_time = time.time()
                    print("--------------------------------------------------------")
                    t += 1
                    if done:
                        break





































































