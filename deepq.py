import tensorflow as tf
import numpy as np
import os
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from replay_buffer import ReplayBuffer
from deepq_learner import DEEPQ

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def evaluate_max_steps(model, eval_env, env_id):  # 评价 18000 个时间步
    episode_score = []
    eval_env.unwrapped.reset()
    obs = np.array(eval_env.reset())
    for step in range(108000):
        action = model.step(tf.constant(obs[None]), stochastic=True, update_eps=0.05)
        obs, rew, done, info = eval_env.step(action)
        if eval_env.was_real_done:  # real done 代表真实的结束, 而不是 lost one life
            score = float(info["episode"]['r'])
            episode_score.append(score)
        if done:
            obs, done = eval_env.reset(), False
    if len(episode_score) == 0:
        return -1000000., 0., 0
    return np.mean(episode_score), np.std(episode_score), len(episode_score)


def learn(env,
          eval_env,
          beta,
          env_id,
          double_q=False,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=int(1e5),
          checkpoint_path="save/",
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          param_noise=False,
          load_path=None):
    # Create all the functions necessary to train the model
    set_global_seeds(seed)

    optimizer = tf.train.AdamOptimizer(lr)

    model = DEEPQ(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        beta=beta,
        grad_norm_clipping=10,
        gamma=gamma,
        double_q=double_q,
        param_noise=param_noise,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    if load_path is not None:
        obs = tf.constant(np.array(env.reset()))
        _ = model.q_network(obs[None])  # shape = (None, n_actions)
        _ = model.target_q_network(obs[None])  # shape = (None, n_actions)
        model.q_network.load_weights(load_path)
        print("Restoring from {}".format(load_path))

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    model.update_target()

    episode_rewards = [0.0]
    episode_scores = []
    evaluate_scores = []
    saved_mean_reward = None
    obs = np.array(env.reset())

    for t in range(total_timesteps):
        update_eps = tf.constant(exploration.value(t))
        action = model.step(tf.constant(obs[None]), update_eps=update_eps)
        new_obs, rew, done, info = env.step(action)
        episode_rewards[-1] += rew

        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, done)
        obs = np.array(new_obs)

        if env.was_real_done:
            episode_scores.append(float(info["episode"]['r']))
        if done:
            obs = np.array(env.reset())
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            loss = model.train_ebu(replay_buffer)

        # Update target network periodically.
        if t > learning_starts and t % target_network_update_freq == 0:
            model.update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        mean_10ep_score = round(np.mean(episode_scores[-11:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("% complete", (t / total_timesteps) * 100)
            logger.record_tabular("steps", t)
            logger.record_tabular("exploring-rate", exploration.value(t))
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("mean 10  episode score", mean_10ep_score)
            logger.dump_tabular()

        # save model.
        if (checkpoint_freq is not None and t > learning_starts and
                num_episodes > 100 and t % checkpoint_freq == 0):
            evaluate_score, evaluate_std, evaluate_len = evaluate_max_steps(model, eval_env, env_id)
            with open(os.path.join(checkpoint_path, 'evaluate_score.txt'), "a+") as fe:
                fe.write("frames: " + str(int(t * 4)) + ", eva-episodes: " + str(evaluate_len) +
                         ", score-std: " + str(round(evaluate_std, 4)) + ", score-mean: " + str(round(evaluate_score, 4)) + "\n")
            evaluate_scores.append([t, evaluate_score, evaluate_std])
            if saved_mean_reward is None or evaluate_score > saved_mean_reward:
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                        saved_mean_reward, evaluate_score))
                if t <= int(total_timesteps / 2):
                    model.q_network.save_weights(os.path.join(checkpoint_path, "model_best_10M.h5"))
                if t > int(total_timesteps / 2):
                    model.q_network.save_weights(os.path.join(checkpoint_path, "model_best_20M.h5"))
                saved_mean_reward = evaluate_score

        if t == int(total_timesteps / 2):
            model.q_network.save_weights(os.path.join(checkpoint_path, "model_10M.h5"))
            with open(os.path.join(checkpoint_path, 'evaluate_score.txt'), "a+") as fe:
                fe.write("Best score 10M: " + str(np.max(np.array(evaluate_scores)[:, 1]))+"\n")

    if total_timesteps > 0:
        with open(os.path.join(checkpoint_path, 'evaluate_score.txt'), "a+") as fe:
            fe.write("Best score 20M: " + str(np.max(np.array(evaluate_scores)[:, 1])))
        np.save(os.path.join(checkpoint_path, "evaluate_scores.npy"), np.array(evaluate_scores))
    return model
