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
    ここでは具体的な方策学習の実装を見ていく

    - 学習は主に以下の手順で進む
    1. 環境と方策から次のバッチを取得する
    2. 取得データの損失計算を行い逆伝播
    3. それに基づき重み最適化。
    4. 2,3のループを１バッチあたりの訓練回数分繰り返す
    """)
    return


@app.cell
def _():
    from torchrl.envs import GymEnv

    env = GymEnv("Pendulum-v1")

    from torchrl.modules import Actor, MLP, ValueOperator
    from torchrl.objectives import DDPGLoss

    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    actor = Actor(MLP(in_features=n_obs,out_features=n_act, num_cells=[32,32]))
    value_net = ValueOperator(
        MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
        in_keys=["observation", "action"],
    )

    ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)
    return actor, ddpg_loss, env


@app.cell
def _(actor, ddpg_loss, env):
    rollout = env.rollout(max_steps=100, policy=actor)
    loss_vals = ddpg_loss(rollout)
    print(loss_vals)
    return (loss_vals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    損失関数の出力は多岐にわたる。特別な重み付けをしないならばこれらをまとめたトータルな損失を計算するとよい
    """)
    return


@app.cell
def _(loss_vals):
    total_loss = 0
    for key, val in loss_vals.items():
        if key.startswith("loss_"):
            total_loss += val
    print(total_loss)
    return (total_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    損失計算の次はそれを用いたバックプロパゲーションと重み更新である
    """)
    return


@app.cell
def _(ddpg_loss, total_loss):
    from torch.optim import Adam
    optim = Adam(ddpg_loss.parameters())
    total_loss.backward()
    optim.step()
    optim.zero_grad()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    一回の更新ごとに全重みが変わっては学習が不安定になる。そこでターゲットNNを用意し、それを割合的に変更していく方法をとる（PPOでは考慮する必要はない。world modelの場合はこういうのが大事になるっぽい）
    """)
    return


@app.cell
def _(ddpg_loss):
    from torchrl.objectives import SoftUpdate
    updater = SoftUpdate(ddpg_loss, eps=0.99)
    return


if __name__ == "__main__":
    app.run()
