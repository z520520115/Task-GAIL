from dm_control import suite
from dm_control.suite.wrappers import pixels
from moviepy.editor import ImageSequenceClip
import numpy as np


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

def main(args):

    # Load one task:
    env = suite.load(domain_name=args.domain_name, task_name=args.task_name)

    # Wrap the environment to obtain the pixels
    env = pixels.Wrapper(env, pixels_only=False)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()
    observation_matrix = []

    while not time_step.last():
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        time_step = env.step(action)
        observation_dm = time_step.observation["pixels"]
        observation_matrix.append(observation_dm)

    clip = ImageSequenceClip(observation_matrix, fps=50)
    clip.write_gif('./test.gif')


if __name__ == '__main__':
    arg = dm_control_parser().parse_args()
    main(arg)