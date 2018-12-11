from mlagents.envs import UnityEnvironment
import numpy as np 
import pickle, os, gzip
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description='Generate Unity scenes (internal brain).')
parser.add_argument('--env', type=str, help='path to game build')
parser.add_argument('--output_dir', type=str, help='location of output data')
parser.add_argument('--n', type=int, default=1000, help='number of episodes')
parser.add_argument('--nstart', type=int, default=0, help='number to start from')
parser.add_argument('--print_every', type=int, default=100, help='print progress')
args = parser.parse_args()

env = UnityEnvironment(file_name=args.env)
# Set the default brain to work with
default_brain = env.brain_names[0]

train_mode = True 

env_info = env.reset(train_mode=train_mode)[default_brain]

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

count = args.nstart
batch_size = len(env_info.agents)
to_store = [defaultdict(list) for _ in range(batch_size)]

last_count = -1
while count < args.n:
    if (count) % args.print_every == 0 and count != last_count:
        print(count)
        last_count = count
    env_info = env.step()[default_brain]
    for i in range(batch_size):
        to_store[i]['previous_action'].append(env_info.previous_vector_actions[i])
        to_store[i]['reward'].append(env_info.rewards[i])
        to_store[i]['vector_observation'].append(env_info.vector_observations[i])
        to_store[i]['visual_observation'].append(env_info.visual_observations[0][i])
        if env_info.local_done[i]:
            count += 1
            with gzip.open(os.path.join(args.output_dir, str(count)+'.p.gz'), "wb") as f:
                pickle.dump(to_store[i], f)
            to_store[i] = defaultdict(list)

print("Done!")