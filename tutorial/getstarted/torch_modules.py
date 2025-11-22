import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch.multiprocessing as mp
    import sys

    mp.set_start_method('spawn',force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Torch modules

    - 環境がTensorDict形式のメタデータとして出力されていたのと同様に、エージェントの方策などにも同形式が用いられる
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from tensordict.nn import TensorDictModule
    from torchrl.envs import GymEnv

    env = GymEnv("Pendulum-v1")
    # 現状学習をしていないのでランダムな重みでのfeed forwardな状態
    module = nn.LazyLinear(out_features=env.action_spec.shape[-1]) 
    policy = TensorDictModule(
        module, 
        in_keys=["observation"], 
        out_keys=["action"],
    )
    return TensorDictModule, env, module, nn, policy


@app.cell
def _(mo):
    mo.md(r"""
    - すべてがTensorDict形式なため、方策学習のための入出力も方策を定義するときに環境の辞書キーを設定することで定義される
    """)
    return


@app.cell
def _(env, policy):
    print(env.action_spec.shape[-1])
    print("------")
    rollout = env.rollout(max_steps=10,policy=policy)
    print(rollout)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    方策として`Actor`を使ってみる

    - 現状として中身は上と変わらないが、現時点ではある種のラベル付けという役割
    """)
    return


@app.cell
def _(env, module):
    from torchrl.modules import Actor

    policy_ = Actor(module)
    rollout_ = env.rollout(max_steps=10,policy=policy_)
    print(rollout_)
    return (Actor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ネットワーク

    - 単なる`Linear`以外の`MLP`や`CNN`モデルにも対応している
    """)
    return


@app.cell
def _(Actor, env, nn):
    from torchrl.modules import MLP

    module_ = MLP(
        out_features=env.action_spec.shape[-1], 
        num_cells=[32,64], 
        activation_class=nn.Tanh, 
    )
    policy__ = Actor(module_)
    rollout__ = env.rollout(max_steps=10,policy=policy__)
    print(rollout__)
    return (MLP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    確率的な方策を用いる場合の実装

    - 確率的な方策を返すためには入力された観測(今回は3次元)を行動の分布のパラメータ(mean,std)として出力するネットワークを作れば良い
    """)
    return


@app.cell
def _(MLP, TensorDictModule, env, nn):
    from tensordict.nn.distributions import NormalParamExtractor
    from torch.distributions import Normal
    from torchrl.modules import ProbabilisticActor

    # 観測から分布のパラメータを得るMLP
    backbone = MLP(in_features=3, out_features=2) # パラメータはこのとき単一のテンソルに２つが格納されてる
    extractor = NormalParamExtractor() # ここでそれぞれ変数が分離して扱われる
    module__ = nn.Sequential(backbone,extractor)
    td_module = TensorDictModule(module__,in_keys=["observation"],out_keys=["loc","scale"])
    policy___ = ProbabilisticActor(
        td_module, 
        in_keys=["loc","scale"], 
        out_keys=["action"], 
        distribution_class = Normal, 
        return_log_prob=True, 
    )

    rollout___ = env.rollout(max_steps=10,policy=policy___)
    print(rollout___)
    return (policy___,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    貪欲にただ方策の最も確率の高いものを選ぶのか、それとも方策の分布からサンプリングしてランダム性を与えるのかは以下のように切り替えることができる
    """)
    return


@app.cell
def _(env, policy___):
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout____ = env.rollout(max_steps=10,policy=policy___)
    with set_exploration_type(ExplorationType.RANDOM):
        rollout____ = env.rollout(max_steps=10,policy=policy___)
    return ExplorationType, set_exploration_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ランダムな方策選択を決定論的に実装する場合は以下
    """)
    return


@app.cell
def _(Actor, ExplorationType, MLP, env, set_exploration_type):
    from tensordict.nn import TensorDictSequential
    from torchrl.modules import EGreedyModule

    policy____ = Actor(MLP(3, 1, num_cells=[32, 64]))
    exploration_module = EGreedyModule(
        spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5
    )
    exploration_policy = TensorDictSequential(policy____, exploration_module)

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        # Turns off exploration
        rollout = env.rollout(max_steps=10, policy=exploration_policy)
    with set_exploration_type(ExplorationType.RANDOM):
        # Turns on exploration
        rollout = env.rollout(max_steps=10, policy=exploration_policy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    チュートリアルではその後行動価値関数の実装へとなるが、現状使う予定はないので省略
    """)
    return


if __name__ == "__main__":
    app.run()
