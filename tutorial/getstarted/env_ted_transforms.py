import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Linux用のmultiprocessingの設定
    import torch.multiprocessing as mp
    import sys
    mp.set_start_method('spawn',force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    環境の作成
    """)
    return


@app.cell
def _():

    from torchrl.envs import GymEnv

    env = GymEnv("Pendulum-v1")
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    環境を動かす

    - TorchRLでは、環境の状態はTensorDict形式で出力される
    - `reset()`: 環境の初期状態を返す
    """)
    return


@app.cell
def _(env):
    reset = env.reset()
    print(reset) 
    return (reset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `rand_action(reset)`: 作成した状態に対してランダムな行動を実行
    - `reset()`が初期状態、`rand_action()`で次の遷移にかかる行動`action`がセットされる
    """)
    return


@app.cell
def _(env, reset):
    reset_with_action = env.rand_action(reset) # 初期状態に何らかの行動を追加
    print(reset_with_action)
    print("------")
    print(reset_with_action["action"])
    return (reset_with_action,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `step()`: セットされた行動を実行。次の状態`next`を作成
    """)
    return


@app.cell
def _(env, reset_with_action):
    stepped_data = env.step(reset_with_action)
    print(stepped_data)
    return (stepped_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `next`の状態に遷移するには`step_mdp()`などを用いる
    - つまり、`reset()`→`rand_action()`など→`step()`→`step_mdp()`→`rand_action()`などというのがrolloutの流れ
    """)
    return


@app.cell
def _(stepped_data):
    from torchrl.envs import step_mdp

    data = step_mdp(stepped_data)
    print(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    環境のrollout

    - 上で見た一連のループをまとめてやってくれるのが`rottlout()`である
    """)
    return


@app.cell
def _(env):
    rollout = env.rollout(max_steps=10)
    print(rollout)
    return (rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - rolloutでは過去のstepもテンソルとして保存されているので過去の状態を見ることができる
    """)
    return


@app.cell
def _(rollout):
    transition = rollout[3]
    print(transition)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    環境の変換

    - 環境情報の変換は、出力のTensorDictの形式を変更することで行う
    - ステップ数を記憶するように変換した例が以下
    """)
    return


@app.cell
def _(env):
    from torchrl.envs import StepCounter, TransformedEnv

    transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
    rollout_ = transformed_env.rollout(max_steps=100)
    print(rollout_)
    return (rollout_,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - StepCounterとrolloutのmax_stepsが異なる場合、少ない方に合わせられる。その場合、途中でrolloutが打ち切られたとみなされ、trancatedがtrueを返す
    """)
    return


@app.cell
def _(rollout_):
    print(rollout_["next","truncated"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最後にmeltingpot版の環境についても見ておく
    """)
    return


@app.cell
def _():
    from torchrl.envs import MeltingpotEnv
    env_ = MeltingpotEnv("commons_harvest__open")
    rollout__ = env_.rollout(max_steps=10)
    print(rollout__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 特筆すべきは、MARL環境なので、単にマップ環境の中に観測があるのではなく、環境の中にエージェントごとの観測や報酬があるという点
    """)
    return


if __name__ == "__main__":
    app.run()
