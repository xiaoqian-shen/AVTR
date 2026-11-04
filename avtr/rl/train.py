"""Launch AVTR adaptive reasoning with verl.

Usage: python -m avtr.rl.train --config-path scripts/configs --config-name rl [key=value ...]
"""

from __future__ import annotations

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging

# Imported by every worker (actor_rollout_ref.model.external_lib): fp32 rotary frequencies, the reference run's
# video and image decoding and the Hugging Face export under the model's own parameter names (runtime), and the
# actor engine that scores the gate tokens over their pair (gate_logprob).
WORKER_LIBS = ("avtr.rl.runtime", "avtr.rl.gate_logprob")


@ray.remote
class TaskRunner:
    """verl's ``TaskRunnerV1`` with this stage's trainer in place of the registered one: ``ImagineRLTrainer``,
    or ``OneStepOffTrainer`` with ``avtr.one_step_off``."""

    def run(self, config: DictConfig):
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        from .one_step_off import OneStepOffTrainer
        from .trainer import ImagineRLTrainer

        configure_verl_logging()
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer, succeeded = None, False
        try:
            trainer = (OneStepOffTrainer if config.avtr.one_step_off else ImagineRLTrainer)(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
                                                teacher_client=trainer.get_teacher_client(),
                                                reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                if getattr(trainer, "logger", None) is not None:
                    trainer.logger.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def check_mini_batch(config: DictConfig) -> None:
    """Reject at launch a GPU count verl only rejects at the first actor update: each PPO mini-batch
    (``ppo_mini_batch_size`` x ``rollout.n`` rows) is split evenly over the data-parallel ranks."""
    actor = config.actor_rollout_ref.actor
    ranks = config.trainer.n_gpus_per_node * config.trainer.nnodes // actor.fsdp_config.ulysses_sequence_parallel_size
    rows = actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n
    if rows % ranks:
        raise ValueError(f"a PPO mini-batch of {rows} rows (actor.ppo_mini_batch_size x rollout.n) does not split "
                         f"evenly over {ranks} data-parallel GPUs")


def check_worker_libs(config: DictConfig) -> None:
    libs = config.actor_rollout_ref.model.get("external_lib") or []
    libs = [libs] if isinstance(libs, str) else list(libs)
    if any(lib not in libs for lib in WORKER_LIBS) or config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError(f"keep {', '.join(WORKER_LIBS)} in actor_rollout_ref.model.external_lib and actor.strategy "
                         "fsdp or fsdp2: the gate tokens are scored over their two-token pair by an FSDP actor engine")


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig) -> None:
    if config.trainer.v1.trainer_mode != "sync":
        raise ValueError("the RL stage builds on verl's sync trainer (trainer.v1.trainer_mode=sync); "
                         "avtr.one_step_off moves the rollout one step behind the update")
    check_worker_libs(config)
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    check_mini_batch(config)
    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
