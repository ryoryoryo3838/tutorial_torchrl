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
    ## メイン
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from tensordict.nn import TensorDictModule
    import meltingpot as mp
    from torchrl.envs import MeltingpotEnv, TransformedEnv, CatTensors, DoubleToFloat, ToTensorImage
    from torchrl.modules import ProbabilisticActor

    ### 環境設定
    env = MeltingpotEnv("commons_harvest__open")
    env = TransformedEnv(env)

    # 画像前処理
    env.append_transform(
        ToTensorImage(
            in_keys=[("agents", "observation", "RGB")],
        )
    )
    # 数値データ前処理＆データ結合
    env.append_transform(
        DoubleToFloat(    
            in_keys = [
                ("agents","observation", "READY_TO_SHOOT"),
                ("agents","observation", "COLLECTIVE_REWARD"),
            ],

        )
    )
    env.append_transform(
        CatTensors(
            in_keys = [
                ("agents","observation", "READY_TO_SHOOT"),
                ("agents","observation", "COLLECTIVE_REWARD"),
            ],
            out_key = ("agents","observation", "vector_obs"),
            dim=-1
        )
    )

    ### アクションの初設定
    action_spec = env.action_spec[("agents","action")]
    if action_spec.shape == torch.Size([]):
        out_features = action_spec.space.n
    else:
        out_features = action_spec.shape[-1]

    module = nn.LazyLinear(out_features=out_features)
    base_model = TensorDictModule(
        module=module,
        in_keys=[("agents","observation","vector_obs")],
        out_keys=[("agents","logits")],
    )

    ### マルチモーダルネットワーク作成
    class MultimodalNet(nn.Module):
        def __init__(self, action_dim=1):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3,32,kernel_size=8,stride=4),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.LazyLinear(256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
        def forward(self,image,vector):
            img_feat = self.cnn(image)
            combined = torch.cat([img_feat,vector],dim=-1)
            logits = self.head(combined)
            return logits
    net = MultimodalNet(action_dim=out_features)

    base_model = TensorDictModule(
        module=net,
        in_keys=[
            ("agents","observation","RGB"),
            ("agents","observation","vector_obs"),
        ],
        out_keys=[("agents","logits")],
    )

    policy = ProbabilisticActor(
        module=base_model,
        in_keys = [("agents","logits")],
        out_keys = [("agents","action")],
        spec=None,
        distribution_class=Categorical,
        return_log_prob=False,
    )
    rollout = env.rollout(max_steps=10, policy=policy)
    print(rollout)
    return MultimodalNet, env, policy, rollout, torch


@app.cell
def _(MultimodalNet):
    ## Critic net
    from torchrl.modules import ValueOperator
    critic_net = MultimodalNet()
    critic_module = ValueOperator(
        module=critic_net,
        in_keys=[
            ("agents","observation","RGB"),
            ("agents","observation","vector_obs"),
        ],
        out_keys=[("agents","state_value")],
    )
    return (critic_module,)


@app.cell
def _(critic_module, env, policy):
    ## train components
    from torchrl.collectors import SyncDataCollector
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE
    from torch.optim import Adam

    collector = SyncDataCollector(
        env,
        policy, 
        frames_per_batch=500, 
        total_frames=100000, 
        split_trajs=False, 
        device="cuda",
    )

    advantage_module = GAE(
        gamma=0.99, 
        lmbda=0.95, 
        value_network=critic_module, 
        device="cuda",
    )

    loss_module = ClipPPOLoss(
        actor_network=policy, 
        critic_network=critic_module, 
        clip_epsilon=0.2, 
        entropy_coef=0.01, 
        normalize_advantage=True,
    )

    optimizer = Adam(loss_module.parameters(), lr=3e-4)

    return advantage_module, collector, loss_module, optimizer


@app.cell
def _(advantage_module, collector, loss_module, optimizer, torch):
    ## Train

    from tqdm.notebook import tqdm 
    logs = {
        "reward": [], 
        "step": []
    }

    pbar = tqdm(total=collector.total_frames)

    for i, tensordict_data in enumerate(collector):
        with torch.no_grad():
            advantage_module(
                tensordict_data,
                value_key=("agents","state_value"),
                reward_key=("agents","reward"), 
            
            )
        data_view = tensordict_data.view(-1)

        for _ in range(4):
            loss_vals = loss_module(data_view)
            loss_value = (
                loss_vals["loss_objective"] 
                + loss_vals["loss_critic"] 
                + loss_vals["loss_entropy"]
            )
            optimizer.zero_grad() 
            loss_value.backward()
            optimizer.step()
        avg_reward = tensordict_data["next","agents","reward"].mean().item()
        logs["reward"].append(avg_reward)
        logs["step"].append(i)

        pbar.update(tensordict_data.numel())
        pbar.set_description(f"reward: {avg_reward:.4f}")

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 描画用コード
    """)
    return


@app.cell
def _(mo, rollout, torch):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML
    import numpy as np

    def save_rollout_git(rollout_tensordict,filename="rollout.git",fps=10,player=-1):
        keys = rollout_tensordict.keys(include_nested=True)
        if "RGB" in keys and player==-1:
            video_data = rollout_tensordict["RGB"]
        elif ("agents", "observation","RGB") in keys and player != -1:
            video_data = rollout_tensordict["agents","observation","RGB"][:,player]
        else:
            "Error"
            return

        video_data = video_data.cpu()
        if video_data.shape[-3] == 3:
            video_data = video_data.permute(0,2,3,1)
        if video_data.dtype in (torch.float32, torch.float64):
            if video_data.max() > 1.0:
                video_data = video_data.byte()
            else:
                video_data = (video_data * 255).byte()
        else:
            video_data = video_data.byte()
        video_data = video_data.numpy()

        # --- デバッグ用出力 ---
        print(f"Video Data Shape: {video_data.shape}")
        print(f"Video Data Range: {video_data.min()} - {video_data.max()}")
        print(f"Video Data Type:  {video_data.dtype}")

        from PIL import Image
        imgs = []
        for frame in video_data:
            img = Image.fromarray(frame)
            new_size = (int(img.width * 4), int(img.height * 4))
            img = img.resize(new_size, resample=Image.NEAREST)
            imgs.append(img)

        imgs[0].save(
            filename, 
            save_all=True, 
            append_images=imgs[1:], 
            duration=1000/fps, 
            loop=0
        )
        print(f"Saved to {filename}")
        return mo.image(filename)

    save_rollout_git(rollout,filename="rollout.gif",fps=10, player=-1) 
    return


@app.cell
def _(mo):
    def display_grid(videos, cols=4, glb=None):
        rows = []
        if glb != None:
            rows.append(
                mo.vstack([
                    mo.md("Global View"),
                    mo.image(glb)
                ])
            )
        for i in range(0,len(videos),cols):
            chunk= videos[i:i+cols]
            row_items = []
            for f in chunk:
                row_items.append(
                    mo.vstack([
                        mo.md(f"Agent {f}"),
                        mo.image(f)
                    ], align="center")
                )
            rows.append(
                mo.hstack(row_items, justify="start", gap=1)
            )
        return mo.vstack(rows, gap=1)
    glb = "rollout.gif"
    videos = [f"rollout{i}.gif" for i in range(6)]
    display_grid(videos,cols=3,glb=glb)
    return


if __name__ == "__main__":
    app.run()
