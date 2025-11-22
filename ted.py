import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TorchRL

    ## Get started with environments, TED and transforms
    """)
    return


@app.cell
def _():
    from torchrl.envs import MeltingpotEnv
    import meltingpot as mp
    env = MeltingpotEnv("commons_harvest__open")
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    あらかじめ公式が有名な環境のwrapperを提供している
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
    環境は、行動を入力として受取、エージェントに観測を与えメタデータを出力するものとして辞書形式で捉えられる

    環境`env`で重要なメソッドが`reset()`と`step()`である
    - `reset()`は環境の初期化を行い、初期観測を返す
    - `step(action)`は環境に行動`action`を与え、次の観測、報酬、終了情報などを返す)
    """)
    return


@app.cell
def _(env, reset):
    reset_with_action = env.rand_action(reset)
    print(reset_with_action)
    print("-------")
    print(reset_with_action["agents"][0]["action"])
    return (reset_with_action,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `MeltingPot`環境における環境の出力メタデータとはどのようなものか
    `reset()`で与えられる初期観測を分解して見ていく

    - fields
        - RGB: マップの全体の各ピクセルを格納したテンソル
        - agents: 各エージェントの環境への情報
            - fields
                - action: 環境へ作用できる行動のリスト（4方向移動＋方向転換+ビーム？）
                - observation: エージェントの知りうる環境の情報
                    - fields:
                        - COLLECTIVE_REWARD: 収集してきた報酬
                        - READY_TO_SHOOT: ビームを打てるか否か（リロードとかの問題？）
                        - RGB: エージェントの観測した環境マップ？（にしては少ないか？）
                    - batch_size
                    - device
                    - is_shared
            - batch_size
            - device
            - is_shared
        - done: 有限ホライゾンであった場合の終了判定
        - next: 初期状態以外で登場する。
            - fields: ~
        - terminated: これもわからない
    - batch_size
    - device
    - is_shared
    """)
    return


@app.cell
def _(reset):
    print(reset["agents"][0])
    return


@app.cell
def _(env, reset_with_action):
    stepped_data = env.step(reset_with_action)
    print(stepped_data)
    return (stepped_data,)


@app.cell
def _(stepped_data):
    from torchrl.envs import step_mdp

    data = step_mdp(stepped_data)
    print(stepped_data["agents"][0]["action"])
    print(data["agents"][0]["action"])
    return


@app.cell
def _(env):
    rollout = env.rollout(max_steps=10)
    print(rollout)
    return


@app.cell
def _(env):
    from torchrl.envs import StepCounter, TransformedEnv
    transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
    rollout_ = transformed_env.rollout(max_steps=100)
    print(rollout_)
    print(rollout_["step_count"])
    return (rollout_,)


@app.cell
def _(rollout_):
    print(rollout_["next", "truncated"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
