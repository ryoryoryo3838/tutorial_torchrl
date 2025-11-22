import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch.multiprocessing as mp
    import sys
    mp.set_start_method('spawn',force=True)
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    学習データや結果を貯蔵する方法についてみていく
    """)
    return


@app.cell
def _():
    import tempfile
    import torch
    from torchrl.collectors import SyncDataCollector
    from torchrl.envs import GymEnv
    from torchrl.envs.utils import RandomPolicy

    torch.manual_seed(0)
    env = GymEnv("CartPole-v1")
    env.set_seed(0)

    policy = RandomPolicy(env.action_spec)
    collector = SyncDataCollector(env,policy, frames_per_batch=200, total_frames=-1 )
    return collector, tempfile


@app.cell
def _(collector):
    for data in collector:
        print(data)
        break
    print("------")
    print(data["collector","traj_ids"])
    return


@app.cell
def _(tempfile):
    from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer

    buffer_scratch_dir = tempfile.TemporaryDirectory().name 
    buffer = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=1000, scratch_dir=buffer_scratch_dir)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
