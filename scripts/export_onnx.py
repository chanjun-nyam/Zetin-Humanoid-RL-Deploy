import typer

app = typer.Typer()



@app.command()
def export_onnx(load_path: str, save_path: str, n_obs: int, n_action: int):
    import torch
    from simple_rl.env import EnvSpec
    from simple_rl.algorithms.ppo import PPO, PPOCfg
    from simple_rl.modules.modules import MlpActorCritic

    # load simple_rl MlpActorCritic model
    env_spec = EnvSpec(
        n_env=1,
        device=torch.device('cpu'),
        n_obs=n_obs,
        n_action=n_action,
    )
    actor_critic = MlpActorCritic(
        n_obs=env_spec.n_obs,
        n_action=env_spec.n_action,
        init_std=1.0,
        net_arch=[512, 256, 128],
        activ_fn=torch.nn.ReLU,
    )
    ppo_cfg = PPOCfg(
        n_rollout=50,
        n_epoch=5,
        n_minibatch=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        desired_kl=0.01,
        normalize_observation=False,
        ratio_clip_param=0.2,
        value_clip_param=0.2,
        grad_norm_clip=1.0,
        normalize_advantage=True,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.0,
    )
    ppo = PPO(env_spec, actor_critic, ppo_cfg)
    ppo.load(load_path)

    # dummy model class
    class DummyModel(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor_critic = actor_critic

        def forward(self, obs):
            return self.actor_critic.policy.compute(obs).mean

    # dummy model
    dummy_model = DummyModel(actor_critic)
    dummy_model.eval()

    # dummy input tensor
    dummy_input = torch.randn(env_spec.n_env, env_spec.n_obs, device=env_spec.device)

    # export pytorch model to onnx model
    torch.onnx.export(
        dummy_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        dynamo=False,
    )



if __name__ == '__main__':
    app()
