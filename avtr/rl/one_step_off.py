"""Overlap next-batch rollouts with the current PPO update on a dedicated rollout pool."""

from __future__ import annotations

import threading
from concurrent.futures import Future

from verl.checkpoint_engine import CheckpointEngineManager, NCCLCheckpointEngine
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager

from .trainer import ImagineRLTrainer


def in_background(fn, *args, **kwargs) -> Future:
    """``fn(*args, **kwargs)`` on a daemon thread, which cannot keep a failed run from exiting."""
    future: Future = Future()

    def run():
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)

    threading.Thread(target=run, name="avtr-next-batch", daemon=True).start()
    return future


class NextBatch:
    """The trainer's replay buffer, drawing each train batch one step ahead: ``draw`` starts the draw in the
    background, the next ``sample`` of the train partition returns its result. The buffer must return only once
    nothing it dispatched is still generating, as verl's sync ReplayBuffer does."""

    def __init__(self, buffer):
        self.buffer = buffer
        self.pending: Future | None = None

    def draw(self, global_steps: int, batch_size: int) -> None:
        self.pending = in_background(self.buffer.sample, global_steps=global_steps, partition_id="train",
                                     batch_size=batch_size)

    def wait(self) -> None:
        if self.pending is not None:
            self.pending.result()

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        if partition_id != "train":
            return self.buffer.sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)
        if self.pending is None:
            raise RuntimeError("no train batch was drawn ahead of this step")
        pending, self.pending = self.pending, None
        return pending.result()


class OneStepOffTrainer(ImagineRLTrainer):
    """``ImagineRLTrainer`` with the rollout on its own GPUs, one step behind the update.

    Step k: submit the prompts of step k+1 and start drawing them (``on_sample_end``), train on batch k, wait
    for batch k+1 (``step``), then send the weights to the rollout pool (``on_step_end``). The trainer GPUs keep
    verl's hybrid vLLM engines asleep; nothing is generated on them.
    """

    def __init__(self, config):
        rollout = config.actor_rollout_ref.rollout
        if rollout.nnodes <= 0:
            raise ValueError("avtr.one_step_off runs vLLM on GPUs of its own: set actor_rollout_ref.rollout.nnodes")
        if rollout.checkpoint_engine.backend == "naive":
            raise ValueError("avtr.one_step_off sends the weights to another pool: set "
                             "actor_rollout_ref.rollout.checkpoint_engine.backend (e.g. nccl)")
        if rollout.checkpoint_engine.backend == "nccl" and NCCLCheckpointEngine is None:
            raise ValueError("avtr.one_step_off: verl's nccl checkpoint engine needs cupy (pip install cupy-cuda12x)")
        super().__init__(config)

    def _build_replay_buffer(self) -> NextBatch:
        return NextBatch(super()._build_replay_buffer())

    def _setup(self):
        super()._setup()
        rollout = self.config.actor_rollout_ref.rollout
        trainer_memory = rollout.gpu_memory_utilization
        rollout.gpu_memory_utilization = rollout.get("standalone_gpu_memory_utilization") or trainer_memory
        try:
            # Created after the trainer pool, whose placement groups take whole nodes: vLLM gets the nodes left.
            self.rollout_servers = LLMServerManager.create(
                config=self.config, start_rank=len(self.llm_server_manager.rollout_replicas))
        finally:
            rollout.gpu_memory_utilization = trainer_memory
        self.rollout_weights = CheckpointEngineManager(
            config=omega_conf_to_dataclass(rollout.checkpoint_engine), actor_wg=self.actor_rollout_wg,
            replicas=self.rollout_servers.get_replicas())

    def get_llm_client(self):
        # A request cut by a weight update is resumed rather than returned truncated (the schedule avoids cuts).
        return self.rollout_servers.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        self.rollout_weights.update_weights(self.global_steps)

    def on_train_begin(self):
        # The first batch is drawn whole before the first step submits the next prompts, which the draw would take in.
        if self.global_steps <= self.total_training_steps:
            super()._add_batch_to_generate()
            self.replay_buffer.draw(self.global_steps, self.config.data.train_batch_size)
            self.replay_buffer.wait()

    def _add_batch_to_generate(self):
        # verl calls this at the start of step k; here it submits the prompts of step k+1.
        if self.global_steps < self.total_training_steps:
            super()._add_batch_to_generate()

    def on_sample_end(self):
        if self.global_steps < self.total_training_steps:
            self.replay_buffer.draw(self.global_steps + 1, self.config.data.train_batch_size)

    def step(self, metrics: dict, timing_raw: dict):
        batch = super().step(metrics, timing_raw)
        with marked_timer("wait_next_batch", timing_raw, color="red"):
            self.replay_buffer.wait()
        return batch

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = self.rollout_weights.update_weights(self.global_steps)
