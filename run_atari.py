from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from deepq import learn
import argparse
import datetime
import os
import re


def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env',  default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--learning-starts', type=int, default=50000)
    parser.add_argument('--target-network-update-freq', type=int, default=10000)
    parser.add_argument('--exploration-fraction', type=float, default=0.05)
    parser.add_argument('--exploration-final-eps', type=float, default=0.1)
    parser.add_argument('--checkpoint-freq', type=int, default=int(1e5))
    parser.add_argument('--double-q', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.5)                        # beta of EBU
    parser.add_argument('--max-episode-steps', type=int, default=None)
    args = parser.parse_args()

    max_episode_steps = args.max_episode_steps
    if args.env in ['ChopperCommandNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4']:
        max_episode_steps = 4500

    # log
    log_dir = os.path.join("result", re.sub("NoFrameskip-v4", "", args.env)+"-EBU-"+datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    env = make_atari(args.env, max_episode_steps=max_episode_steps)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    # env for evaluation
    if not os.path.exists('tmp'):
        os.mkdir("tmp")
    eval_env = make_atari(args.env, max_episode_steps=max_episode_steps)
    eval_env = bench.Monitor(eval_env, "tmp/"+datetime.datetime.now().strftime("%m-%d-%H-%M-%S"), allow_early_resets=True)
    eval_env = wrap_atari_dqn(eval_env)

    model = learn(
        env,
        eval_env,
        beta=args.beta,
        double_q=args.double_q,
        param_noise=False,
        train_freq=4,
        gamma=0.99,
        lr=args.lr,
        total_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        learning_starts=args.learning_starts,
        target_network_update_freq=args.target_network_update_freq,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=log_dir,
        env_id=args.env
    )

    model.save_weights(os.path.join(log_dir, 'model_20M.h5'))
    env.close()


if __name__ == '__main__':
    main()
