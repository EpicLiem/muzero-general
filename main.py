import pathlib

import ray
import wandb

from muzero import MuZero

if __name__ == "__main__":

    muzero = MuZero('spiel')

    try:
        checkpoints = sorted(
            (pathlib.Path("results") / 'spiel').glob("*/")
        )
        checkpoints.reverse()
        checkpoint_path = checkpoints[0] / "model.checkpoint"
        replay_buffer_path = checkpoints[0] / "replay_buffer.pkl"
        print('checkpoint loaded')
        muzero.load_model(
            checkpoint_path=checkpoint_path,
            replay_buffer_path=replay_buffer_path,
        )
        print('model loaded from checkpoint')
    except:
        print('error loading from checkpoint')

    # wandb.tensorboard.patch(root_logdir="results")
    # wandb.init(project='MuChes', sync_tensorboard=True)
    muzero.train()
    ray.shutdown()
