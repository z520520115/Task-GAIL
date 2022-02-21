from dm_control import viewer
import numpy as np
from dm_control import suite
import PPO_dmcontrol

def arg_parser():
  """
      Create an empty argparse.ArgumentParser.
      """
  import argparse
  return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def dm_control_parser():
  """
      Create an argparse.ArgumentParser for DM Control.
      """
  parser = arg_parser()
  parser.add_argument('--domain_name', help='Domain Name', type=str, default='swimmer')
  parser.add_argument('--task_name', help='Task Name', type=str, default='swimmer6')
  parser.add_argument('--num_timesteps', type=int, default=int(1e6))
  parser.add_argument('--use_pixels', help='Use rgb instead of low dim state rep?', type=bool, default=False)
  parser.add_argument('--seed', help='RNG seed', type=int, default=0)
  return parser

def to_state(time_step):
    state = np.concatenate([x for x in time_step.observation.values()])
    return state

def policy(time_step):
    state = to_state(time_step)
    action = PPO_dmcontrol.ppo.select_action(state, None)
    return action

def main(args):
    # Load one task:
    env = suite.load(domain_name=args.domain_name, task_name=args.task_name)
    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()

    while not time_step.last():
        viewer.launch(env, policy=policy)

if __name__ == '__main__':
    arg = dm_control_parser().parse_args()
    main(arg)

# viewer是单步的 不知如何运行在循环中
# 在这个文件调用PPO_dmcontrol文件竟然会进行训练
# 解决报错