import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch.multiprocessing as mp
    import sys
    mp.set_start_method('spawn',force=True)
    return


@app.cell
def _():
    from torchrl.record import CSVLogger
    logger = CSVLogger(exp_name="tutorial_exp")
    logger.log_scalar("my_scalar", 0.4)
    return (logger,)


@app.cell
def _(logger):
    from torchrl.envs import GymEnv
    env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)
    print(env.rollout(max_steps=3))

    from torchrl.envs import TransformedEnv
    from torchrl.record import VideoRecorder

    recorder = VideoRecorder(logger, tag="my_video")
    record_env = TransformedEnv(env, recorder)

    rollout = record_env.rollout(max_steps=3)
    recorder.dump()
    return


if __name__ == "__main__":
    app.run()
