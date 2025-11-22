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
    ## Lib
    """)
    return


app._unparsable_cell(
    r"""
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    from torchrl.envs import MeltingpotEnv, TransformedEnv, ToTensorImage, CatTensors, DoubleToFloat
    from torchrl.modules import ProbabilisticActor, ConvNet

    from torchrl.collectors import SyncDataCollector 
    from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement

    from torchrl.objectives import ClipPPOLoss 
    from

    from tensordict.nn import TensorDictModule
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Env
    """)
    return


@app.cell
def _(CatTensors, DoubleToFloat, MeltingpotEnv, ToTensorImage, TransformedEnv):
    env = MeltingpotEnv("commons_harvest__open")
    env = TransformedEnv(env)
    env.append_transform(
        ToTensorImage(
            in_keys=[("agents","observation","RGB")],
        )
    )
    env.append_transform(
        DoubleToFloat(
            in_keys= [
                ("agents","observation", "READY_TO_SHOOT"),
                ("agents","observation", "COLLECTIVE_REWARD"),
            ], 
        )
    )
    env.append_transform(
        CatTensors(
            in_keys= [
                ("agents","observation", "READY_TO_SHOOT"),
                ("agents","observation", "COLLECTIVE_REWARD"),
            ], 
            out_key= ("agents","observation", "vector_obs"), 
            dim=-1
        )
    )
    return (env,)


@app.cell
def _(env):
    """Debug"""
    print(env.reset())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Act
    """)
    return


@app.cell
def _(env):
    action_spec = env.action_spec[("agents","action")]
    out_features = action_spec.n
    return action_spec, out_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **action_spec**

    - shape: 各エージェントの方策(7人)
    - space: 行動空間(前後左右、時計、半時計、ビーム、とどまるの８つ？)
    """)
    return


@app.cell
def _(action_spec, out_features):
    print(action_spec)
    print(out_features)
    return


@app.cell
def _():
    ## Model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _(nn, torch):
    class MultiCnnNet(nn.Module):
        def __init__(self,action_dim=1):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3,32,kernel_size=8,stride=4),
                nn.Tanh(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),
                nn.Tanh(),
                nn.Conv2d(64,64,kernel_size=3,stride=2),
                nn.Tanh(), 
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.LazyLinear(256), 
                nn.Tanh(), 
                nn.Linear(256, action_dim),
            )
        def forward(self,image,vector): 
            img_feat = self.cnn(image)
            combined = torch.cat([img_feat,vector],dim=-1)
            logits = self.head(combined)
            return logits
    return (MultiCnnNet,)


@app.cell
def _(MultiCnnNet, TensorDictModule, nn, out_features):
    module = nn.LazyLinear(out_features=out_features)
    net = MultiCnnNet(action_dim=out_features)
    base_model = TensorDictModule(
        module=net, 
        in_keys=[
            ("agents","observation","RGB"), 
            ("agents", "observation", "vector_obs"),
        ], 
        out_keys=[("agents","logits")],
    )
    return (base_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy
    """)
    return


@app.cell
def _(Categorical, ProbabilisticActor, base_model):
    policy= ProbabilisticActor(
        module=base_model, 
        in_keys=[("agents","logits")], 
        out_keys=[("agents","actions")], 
        spec=None, 
        distribution_class=Categorical, 
        return_log_prob=False,
    )
    return (policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value network
    """)
    return


@app.cell
def _(MultiCnnNet, TensorDictModule):
    critic_net = MultiCnnNet(action_dim=1)
    value_module = TensorDictModule(
        module=critic_net, 
        in_keys=[
            ("agents","observation","RGB"), 
            ("agents","observation","vector_obs"),
        ], 
        out_keys=[("agents","state_value")],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Collect & Buffer
    """)
    return


@app.cell
def _(
    LazyTensorStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
    SyncDataCollector,
    env,
    policy,
):
    frames_per_batch = 1000
    total_frames=10000

    collector = SyncDataCollector(
        env, 
        policy, 
        frames_per_batch=frames_per_batch, 
        split_trajs=False, # not RNN
        device="cpu"
    )

    replay_buffer= ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch), 
        sampler = SamplerWithoutReplacement() 
    
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss & GAE
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
