# Training

## Stages 0–2

| Stage | Recipe | Trainable components | Initialization |
|---|---|---|---|
| 0 · Reconstruction | [stage0.yaml](../scripts/configs/stage0.yaml) | Latent queries, connector, SANA transformer | Qwen3.5 + SANA |
| 1 · Visual reflection | [stage1.yaml](../scripts/configs/stage1.yaml) | Latent queries, connector, SANA transformer | Stage 0 |
| 2 · SFT | [stage2.yaml](../scripts/configs/stage2.yaml) | VLM, latent queries, markers, connector | Stage 1 |

### Running

```bash
export DATA_ROOT=DATA_PATH NPROC=8
OUTPUT_DIR=STAGE0_OUTPUT_PATH scripts/train.sh 0
INIT_FROM=STAGE0_CKPT_PATH OUTPUT_DIR=STAGE1_OUTPUT_PATH scripts/train.sh 1
INIT_FROM=STAGE1_CKPT_PATH OUTPUT_DIR=STAGE2_OUTPUT_PATH scripts/train.sh 2
```

Append `key.path=value` to override a recipe. Set `BACKBONE` or `SANA` to change the base models.
Use the complete `checkpoint-N` folder as `INIT_FROM`. Reusing `OUTPUT_DIR` resumes the latest
complete checkpoint; `RESUME_FROM` selects one explicitly.

For multiple nodes, run the launcher on each node with a unique `NODE_RANK`, the same
`MASTER_ADDR`, and shared checkpoint/output paths:

```bash
INIT_FROM=STAGE0_CKPT_PATH OUTPUT_DIR=STAGE1_OUTPUT_PATH \
    NNODES=8 NPROC=8 NODE_RANK=0 MASTER_ADDR=node0 scripts/train.sh 1 \
    train.gradient_accumulation_steps=2
```

Adjust `train.gradient_accumulation_steps` when changing GPU count or global batch size.

### HF checkpoint export

After stage 2, bake latent queries and routing markers into an HF-format checkpoint:

```bash
python -m avtr.tools.export --ckpt STAGE2_CKPT_PATH --out HF_CKPT_PATH --emit-markers
```

Use `--emit-markers` for adaptive inference and RL. Token IDs in the tokenizer files and
checkpoint config must agree. The export contains the backbone and query embeddings; SANA is
not needed for inference.

## Adaptive reasoning with RL

RL jointly optimizes temporal grounding and visual/textual route selection.
Settings are in [rl.yaml](../scripts/configs/rl.yaml).

### Running RL

```bash
pip install flash-attn==2.8.3 --no-build-isolation

CKPT=HF_CKPT_PATH RL_DATA=RL_DATA_PATH OUTPUT_DIR=RL_OUTPUT_PATH scripts/rl.sh
```

For on-policy training on one node:

```bash
CKPT=HF_CKPT_PATH RL_DATA=RL_DATA_PATH OUTPUT_DIR=RL_OUTPUT_PATH scripts/rl.sh \
    avtr.one_step_off=false trainer.nnodes=1
```

### Distributed layout

The distributed recipe uses eight training nodes and one rollout node, with eight GPUs per node.
Export the following on every node before starting Ray, using shared paths:

```bash
export CKPT=HF_CKPT_PATH
export RL_DATA=RL_DATA_PATH
export OUTPUT_DIR=RL_OUTPUT_PATH
export AVTR_REPO=AVTR_REPO_PATH
export PYTHONPATH="$AVTR_REPO"
export FORCE_QWENVL_VIDEO_READER=torchcodec
```

Run `scripts/rl.sh` on the Ray head node. Adjust `trainer.nnodes`, `trainer.n_gpus_per_node`, and
`actor_rollout_ref.rollout.nnodes` for other layouts. Smaller runs can enable
`actor_rollout_ref.actor.fsdp_config.offload_policy=true`; this uses host RAM.

Only a small single-node on-policy run has been checked; the distributed overlap configuration
has not yet been validated end to end.

### Checkpoints

RL checkpoints are saved under `OUTPUT_DIR/global_step_N/actor/`; its `huggingface/` directory
can be loaded directly for inference and evaluation. Restarting resumes the latest step.
Logs go to the console and TensorBoard; `TENSORBOARD_DIR` selects the log directory.
