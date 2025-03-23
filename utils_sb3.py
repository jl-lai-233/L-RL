import os
import numpy as np
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import time

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        file_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
        # self.save_path = os.path.join(log_dir, 'best_model')
        # self.save_path = os.path.join(log_dir)
        self.save_path = os.path.join(log_dir + str(file_id) + '_model')     # 文件夹名字  加上时间戳
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))   # 常规保存 model_id

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir),'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.best_save_path = os.path.join(self.save_path, str(model_id) + '_lowest_model')
                  self.model.save(self.best_save_path)

          self.regular_save_path = os.path.join(self.save_path, str(model_id) + '_model')   # 常规保存
          self.model.save(self.regular_save_path)                                           # 常规保存
        return True