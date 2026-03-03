# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""

import argparse
import asyncio
import dataclasses
import importlib
import os
import queue
import socket
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable
from concurrent import futures
from datetime import timedelta
from typing import Any

import aiohttp
import backoff
import datasets
import deepspeed
import openai
import ray
import torch
import torch.distributed
import uvicorn
import vllm
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    ProcessGroup,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core import kv_cache_utils

from open_instruct import logger_utils
from open_instruct.data_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics, ToolCallStats
from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY, RAW_PROMPT_KEY, VERIFIER_SOURCE_KEY
from open_instruct.environments.base import EnvCall, RolloutState, StepResult
from open_instruct.environments.tools.parsers import ToolParser, create_tool_parser
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.utils import ModelDims, get_device_name, ray_get_with_progress

logger = logger_utils.setup_logger(__name__)

NUM_PREFETCH_WORKERS = 2
DRAIN_ACTIVE_TASKS_SLEEP_S = 1
SHOULD_STOP_TIMEOUT_S = 0.1
INFERENCE_INIT_TIMEOUT_S = 1200
VLLM_HEALTH_CHECK_TIMEOUT_S = 600.0
REQUEST_HEALTH_CHECK_ENABLED = os.environ.get("RLVR_VLLM_REQUEST_HEALTH_CHECK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def model_dims_from_vllm_config(vllm_config: "vllm.config.VllmConfig") -> ModelDims:
    model_config = vllm_config.model_config
    hidden_size = model_config.get_hidden_size()
    intermediate_size = getattr(model_config.hf_text_config, "intermediate_size", 4 * hidden_size)
    sliding_window = getattr(model_config.hf_text_config, "sliding_window", None)
    num_layers = model_config.get_num_layers(vllm_config.parallel_config)
    num_sliding_window_layers = 0

    if sliding_window is not None:
        layer_types = getattr(model_config.hf_text_config, "layer_types", None)
        num_sliding_window_layers = layer_types.count("sliding_attention") if layer_types is not None else num_layers

    return ModelDims(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=model_config.get_vocab_size(),
        num_attn_heads=model_config.hf_text_config.num_attention_heads,
        num_kv_heads=model_config.hf_text_config.num_key_value_heads,
        head_dim=model_config.get_head_size(),
        sliding_window=sliding_window,
        num_sliding_window_layers=num_sliding_window_layers,
        device_name=get_device_name(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
    )


@dataclasses.dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    n: int = 1
    stop: list[str] | None = None
    seed: int | None = None
    logprobs: int | None = 1


@dataclasses.dataclass
class CompletionOutput:
    index: int
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    cumulative_logprob: float = 0.0
    mask: list[int] | None = None
    rollout_state: dict = dataclasses.field(default_factory=dict)
    """Rollout state dict — rewards, step_count, done, tool_output, tool_error, etc."""


@dataclasses.dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = True


def process_tool_tokens(
    tool_outputs: list[str],
    tool_parser: ToolParser,
    tokenizer,
    current_prompt_len: int,
    current_response_len: int,
    max_model_len: int,
    max_tokens: int,
    mask_tool_use: bool,
    role: str = "tool",
) -> tuple[list[int], list[float], list[int], int]:
    """Format, tokenize, and truncate tool outputs.

    Args:
        tool_outputs: Raw outputs from tool calls.
        tool_parser: Parser to format tool outputs.
        tokenizer: Tokenizer to encode formatted output.
        current_prompt_len: Current length of the prompt (for truncation).
        current_response_len: Current length of the response (for truncation).
        max_model_len: Maximum model sequence length.
        max_tokens: Maximum response tokens.
        mask_tool_use: Whether to mask tool tokens in loss computation.
        role: Chat role for formatting (e.g. "tool", "user" for text envs).

    Returns:
        Tuple of (tokens, logprobs, masks, excess).
    """
    formatted_output = tool_parser.format_tool_outputs(tool_outputs, role=role)
    tokens = tokenizer.encode(formatted_output, add_special_tokens=False)

    tokens, excess = truncate_tool_output_tokens(
        tokens,
        current_prompt_len=current_prompt_len,
        current_response_len=current_response_len,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
    )

    logprobs = [0.0] * len(tokens)
    masks = [0 if mask_tool_use else 1] * len(tokens)

    return tokens, logprobs, masks, excess


def assert_threaded_actor(instance):
    """Assert that an instance's class is suitable for use in a threaded (non-async) Ray actor.

    This function performs two checks:
      1. The class must not define any `async def` methods
         (including async generators, staticmethods, or classmethods).
      2. There must not be a running asyncio event loop in the current thread.

    Args:
        instance: The instance whose class to inspect.

    Raises:
        AssertionError: If the class defines one or more async methods, or a running asyncio event loop is detected.
    """
    try:
        loop = asyncio.get_running_loop()
        raise AssertionError(
            f"{instance.__class__.__name__} must run in a threaded Ray actor (no running event loop). "
            f"Detected RUNNING loop={loop!r} on thread='{threading.current_thread().name}'. "
            f"Python={sys.version.split()[0]}."
        )
    except RuntimeError:
        return


def truncate_tool_output_tokens(
    tool_output_token_ids: list[int],
    current_prompt_len: int,
    current_response_len: int,
    max_model_len: int,
    max_tokens: int,
) -> tuple[list[int], int]:
    """Truncate tool output tokens to fit within max_model_len and max_tokens.

    Args:
        tool_output_token_ids: Token IDs from the tool output to potentially truncate.
        current_prompt_len: Number of tokens in the current prompt (original + accumulated).
        current_response_len: Number of tokens in the response so far (for max_tokens check).
        max_model_len: Maximum total sequence length the model can handle.
        max_tokens: Maximum number of response tokens allowed.

    Returns:
        A tuple of (truncated_tokens, excess) where excess is the number of tokens
        that exceeded max_model_len (0 if no truncation due to max_model_len).
    """
    total_len = current_prompt_len + len(tool_output_token_ids)
    excess = max(0, total_len - max_model_len)
    if excess > 0:
        tool_output_token_ids = tool_output_token_ids[:-excess] if excess < len(tool_output_token_ids) else []

    remaining = max(0, max_tokens - current_response_len)
    return tool_output_token_ids[:remaining], excess


# Edited from: https://github.com/OpenRLHF/OpenRLHF/pull/971/files
# Turns out Ray doesnt necessarily place bundles together,
# so this function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
# avoids unnecessary communication for TP>1 with vllm.
def get_bundle_indices_list(placement_group: ray.util.placement_group) -> list[int]:
    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = defaultdict(list)
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles[node_id].append(bundle)

    flattened_bundle_indices = []
    for bundles in node_id_to_bundles.values():
        flattened_bundle_indices.extend(bundles)
    return flattened_bundle_indices


def make_request_id(request: PromptRequest) -> str:
    """Generate a unique tracking key for a request."""
    prefix = "eval" if request.is_eval else "train"
    return f"{prefix}_{request.prompt_id}"


def split_request_id(full_request_id: str) -> dict:
    """Split request ID into base ID and request index.

    >>> split_request_id("train_0_43039_0")
    {'base_id': 'train_0_43039', 'request_index': 0}
    >>> split_request_id("eval_0_12345_2")
    {'base_id': 'eval_0_12345', 'request_index': 2}
    """
    parts = full_request_id.split("_")
    return {"base_id": "_".join(parts[:-1]), "request_index": int(parts[-1])}


def process_completed_request(request_id, outs, current_time, use_tools, request_metadata):
    """Process a completed request with all its samples and return the result.

    Args:
        request_id: The base request ID
        outs: List of RequestOutput objects for all sub-requests
        current_time: Current timestamp for performance metrics
        use_tools: Boolean indicating if tools were used
        request_metadata: Dictionary containing metadata for all requests

    Returns:
        Tuple of (result, is_eval) where result is a GenerationResult and is_eval is a boolean
    """
    final_output = RequestOutput(
        request_id=request_id,
        prompt_token_ids=outs[0].prompt_token_ids,
        outputs=[completion for out in outs for completion in out.outputs],
    )

    total_generation_tokens = sum(len(completion.token_ids) for out in outs for completion in out.outputs)
    metadata = request_metadata[request_id]

    response_ids = [list(out.token_ids) for out in final_output.outputs]
    finish_reasons = [out.finish_reason for out in final_output.outputs]

    logprobs = []
    for idx, out in enumerate(final_output.outputs):
        assert len(out.token_ids) == len(out.logprobs), (
            f"CompletionOutput {idx}: token_ids length ({len(out.token_ids)}) != logprobs length ({len(out.logprobs)})"
        )
        logprobs.append(out.logprobs)

    if use_tools:
        rollout_states = [out.rollout_state for out in final_output.outputs]
        masks = [getattr(out, "mask", [1] * len(out.token_ids)) for out in final_output.outputs]
        num_calls = [rs.get("step_count", 0) for rs in rollout_states]
        timeouts = [rs.get("timeout", False) for rs in rollout_states]
        tool_errors = [rs.get("tool_error", "") for rs in rollout_states]
        tool_outputs = [rs.get("tool_output", "") for rs in rollout_states]
        tool_runtimes = [rs.get("tool_runtime", 0.0) for rs in rollout_states]
        tool_calleds = [rs.get("step_count", 0) > 0 for rs in rollout_states]
        tool_call_stats = [[ToolCallStats(**s) for s in rs.get("tool_call_stats", [])] for rs in rollout_states]
    else:
        rollout_states = [{} for _ in response_ids]
        masks = [[1] * len(resp) for resp in response_ids]
        num_calls = [0] * len(response_ids)
        timeouts = [False] * len(response_ids)
        tool_errors = [""] * len(response_ids)
        tool_outputs = [""] * len(response_ids)
        tool_runtimes = [0.0] * len(response_ids)
        tool_calleds = [False] * len(response_ids)
        tool_call_stats = [[] for _ in response_ids]

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=RequestInfo(
            num_calls=num_calls,
            timeouts=timeouts,
            tool_errors=tool_errors,
            tool_outputs=tool_outputs,
            tool_runtimes=tool_runtimes,
            tool_calleds=tool_calleds,
            tool_call_stats=tool_call_stats,
            rollout_states=rollout_states,
        ),
        index=metadata["index"],
        prompt_id=metadata["prompt_id"],
        token_statistics=TokenStatistics(
            num_prompt_tokens=len(metadata["prompt_token_ids"]),
            num_response_tokens=total_generation_tokens,
            generation_time=current_time - metadata["start_time"],
        ),
        start_time=metadata["start_time"],
        logprobs=logprobs,
    )
    return result, metadata["is_eval"]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    def _is_truthy(value: object) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    return any(_is_truthy(env_vars.get(env_var)) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def _patch_vllm_ray_executor_explicit_visible_devices() -> None:
    """Patch vLLM Ray workers with stable RLVR runtime hints before init_device.

    Ray worker wrappers are created before vLLM initializes CUDA. We patch
    `RayDistributedExecutor._get_env_vars_to_be_updated` so worker wrapper
    `update_environment_variables` always receives RLVR patch markers and setup
    hook hints. When explicit rollout GPU visibility is configured
    (`RLVR_VLLM_CUDA_VISIBLE_DEVICES` / `VLLM_CUDA_VISIBLE_DEVICES`), we also
    pass it through as `CUDA_VISIBLE_DEVICES`.
    """
    explicit = (
        os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
        or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()

    try:
        from vllm.platforms import current_platform as vllm_current_platform
    except Exception:
        return

    executor_targets: list[tuple[str, type[Any]]] = []
    seen_ids: set[int] = set()
    for module_path in (
        "vllm.v1.executor.ray_distributed_executor",
        "vllm.executor.ray_distributed_executor",
    ):
        try:
            module = importlib.import_module(module_path)
            executor_cls = getattr(module, "RayDistributedExecutor", None)
            if executor_cls is None:
                continue
            cls_id = id(executor_cls)
            if cls_id in seen_ids:
                continue
            seen_ids.add(cls_id)
            executor_targets.append((module_path, executor_cls))
        except Exception:
            continue

    if not executor_targets:
        return

    device_control_env_var = getattr(vllm_current_platform, "device_control_env_var", None) or "CUDA_VISIBLE_DEVICES"
    ray_noset_key = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    ray_override_key = "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"
    ray_override_value = os.environ.get(ray_override_key, "0")
    ray_noset_value = os.environ.get(ray_noset_key, "1")

    # Ray wraps every actor method with set/reset visible accelerator env.
    # With NOSET enabled, that wrapper still captures then restores the pre-call
    # CUDA_VISIBLE_DEVICES value, which can undo our explicit override.
    # For explicit rollout visibility we bypass that wrapper entirely.
    try:
        import ray._private.utils as ray_private_utils
    except Exception:
        ray_private_utils = None

    if ray_private_utils is not None and not getattr(ray_private_utils, "_rlvr_noset_visible_patch_applied", False):
        original_set_visible_accelerator_ids = getattr(ray_private_utils, "set_visible_accelerator_ids", None)
        if original_set_visible_accelerator_ids is not None:
            def _patched_set_visible_accelerator_ids():
                explicit_local = (
                    os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
                    or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
                    or ""
                ).strip()
                noset_enabled = str(os.environ.get(ray_noset_key, "")).strip().lower() in {"1", "true", "yes", "on"}
                if explicit_local and noset_enabled:
                    return {}
                return original_set_visible_accelerator_ids()

            ray_private_utils.set_visible_accelerator_ids = _patched_set_visible_accelerator_ids
            ray_private_utils._rlvr_noset_visible_patch_applied = True
            logger.info("Patched Ray set_visible_accelerator_ids for explicit rollout visibility.")

    for module_path, executor_cls in executor_targets:
        if getattr(executor_cls, "_rlvr_explicit_visible_patch_applied", False):
            continue

        original_get_envs = getattr(executor_cls, "_get_env_vars_to_be_updated", None)
        original_init_workers_ray = getattr(executor_cls, "_init_workers_ray", None)
        if original_get_envs is None or original_init_workers_ray is None:
            continue

        def _patched_get_env_vars_to_be_updated(self, _orig=original_get_envs):
            envs_list = _orig(self)
            explicit_local = (
                os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
                or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
                or ""
            ).strip()

            for worker_env in envs_list:
                if isinstance(worker_env, dict):
                    worker_env["RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH"] = "1"
                    worker_env["RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH"] = "1"
                    # Keep worker Ray visibility settings aligned with the actor.
                    worker_env[ray_noset_key] = ray_noset_value
                    worker_env[ray_override_key] = ray_override_value
                    if explicit_local:
                        worker_env[device_control_env_var] = explicit_local
                        worker_env["RLVR_VLLM_CUDA_VISIBLE_DEVICES"] = explicit_local
                        worker_env["VLLM_CUDA_VISIBLE_DEVICES"] = explicit_local
                    else:
                        worker_env.pop(device_control_env_var, None)
                        worker_env.pop("RLVR_VLLM_CUDA_VISIBLE_DEVICES", None)
                        worker_env.pop("VLLM_CUDA_VISIBLE_DEVICES", None)

            if explicit_local:
                logger.info(
                    "Overriding vLLM Ray worker %s=%s from explicit visibility hint.",
                    device_control_env_var,
                    explicit_local,
                )
            else:
                logger.info(
                    "Injected RLVR vLLM Ray worker patch markers without explicit %s override.",
                    device_control_env_var,
                )
            return envs_list

        def _patched_init_workers_ray(self, placement_group, _orig=original_init_workers_ray, **ray_remote_kwargs):
            explicit_local = (
                os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
                or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
                or ""
            ).strip()
            runtime_env = ray_remote_kwargs.get("runtime_env")
            if runtime_env is None:
                runtime_env = {}
                ray_remote_kwargs["runtime_env"] = runtime_env
            if isinstance(runtime_env, dict):
                runtime_env.setdefault(
                    "worker_process_setup_hook",
                    "flwr_model.rlvr.bootstrap.patch_open_instruct_utils",
                )
                env_vars = runtime_env.get("env_vars")
                if env_vars is None:
                    env_vars = {}
                    runtime_env["env_vars"] = env_vars
                if isinstance(env_vars, dict):
                    env_vars["RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH"] = "1"
                    env_vars["RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH"] = "1"
                    env_vars.setdefault(ray_noset_key, ray_noset_value)
                    env_vars.setdefault(ray_override_key, ray_override_value)
                    if explicit_local:
                        env_vars[device_control_env_var] = explicit_local
                        env_vars["RLVR_VLLM_CUDA_VISIBLE_DEVICES"] = explicit_local
                        env_vars["VLLM_CUDA_VISIBLE_DEVICES"] = explicit_local
                    else:
                        env_vars.pop(device_control_env_var, None)
                        env_vars.pop("RLVR_VLLM_CUDA_VISIBLE_DEVICES", None)
                        env_vars.pop("VLLM_CUDA_VISIBLE_DEVICES", None)
                    logger.info(
                        "Injected vLLM Ray worker runtime_env hints: setup_hook=%s, explicit_visibility=%s",
                        "flwr_model.rlvr.bootstrap.patch_open_instruct_utils",
                        bool(explicit_local),
                    )
            return _orig(self, placement_group, **ray_remote_kwargs)

        executor_cls._get_env_vars_to_be_updated = _patched_get_env_vars_to_be_updated
        executor_cls._init_workers_ray = _patched_init_workers_ray
        executor_cls._rlvr_explicit_visible_patch_applied = True
        logger.info("Applied explicit rollout visibility patch to %s", module_path)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str | None = None,
    pg_options: Any | None = None,
    device_id: torch.device | int | None = None,
) -> ProcessGroup:
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
        device_id=device_id,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


@backoff.on_exception(backoff.constant, (aiohttp.ClientError, RuntimeError), max_time=60, interval=0.5)
async def _check_health(port: int) -> None:
    async with (
        aiohttp.ClientSession() as session,
        session.get(
            f"http://127.0.0.1:{port}/health", timeout=aiohttp.ClientTimeout(total=VLLM_HEALTH_CHECK_TIMEOUT_S)
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"vLLM server health check failed with status {response.status}")


def _prefetch_worker(actor: "LLMRayActor") -> None:
    while True:
        if actor._should_stop() or len(actor.active_tasks) >= actor.inference_batch_size:
            time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)
            continue

        request = actor.prompt_queue.get()
        add_request(actor, request)


def add_request(actor: "LLMRayActor", request: PromptRequest) -> None:
    request_id = make_request_id(request)
    sampling_params = dataclasses.replace(request.generation_config, n=1)

    actor.request_metadata[request_id] = {
        "is_eval": request.is_eval,
        "index": request.index,
        "prompt_id": request.prompt_id,
        "sampling_params": sampling_params,
        "original_sampling_params": request.generation_config,
        "prompt_token_ids": list(request.prompt),
        "start_time": time.perf_counter(),
        "active_tools": request.active_tools,
        "env_config": request.env_config,
    }

    for j in range(request.generation_config.n):
        seed = request.generation_config.seed + j if request.generation_config.seed is not None else None
        sub_sampling_params = dataclasses.replace(sampling_params, seed=seed)
        sub_request_id = f"{request_id}_{j}"
        actor.active_tasks[sub_request_id] = asyncio.run_coroutine_threadsafe(
            process_request(actor, sub_request_id, sub_sampling_params), actor.loop
        )


FALLBACK_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def _create_server_args(model_path: str, has_chat_template: bool) -> argparse.Namespace:
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    cli_args = ["--model", model_path]
    if not has_chat_template:
        cli_args.extend(["--chat-template", FALLBACK_CHAT_TEMPLATE])
    args = parser.parse_args(cli_args)
    args.disable_fastapi_docs = True
    return args


def accumulate_completions(actor: "LLMRayActor", sub_request: dict) -> futures.Future | None:
    base_request_id = sub_request["base_request_id"]
    expected_n = sub_request["expected_n"]

    if base_request_id not in actor.request_outputs:
        actor.request_outputs[base_request_id] = {
            "outputs": [],
            "expected_n": expected_n,
            "use_tools": sub_request["use_tools"],
        }

    actor.request_outputs[base_request_id]["outputs"].append(sub_request["request_output"])

    if len(actor.request_outputs[base_request_id]["outputs"]) == expected_n:
        return asyncio.run_coroutine_threadsafe(finalize_completed_request(actor, base_request_id), actor.loop)

    return None


async def finalize_completed_request(actor: "LLMRayActor", base_request_id: str) -> None:
    outputs = actor.request_outputs[base_request_id]["outputs"]
    ordered_outs = sorted(outputs, key=lambda x: split_request_id(x.request_id)["request_index"])

    current_time = time.perf_counter()
    result, is_eval = process_completed_request(
        base_request_id,
        ordered_outs,
        current_time,
        actor.request_outputs[base_request_id]["use_tools"],
        actor.request_metadata,
    )

    actor.request_outputs.pop(base_request_id)
    actor.request_metadata.pop(base_request_id, None)

    dataset = actor.eval_dataset if is_eval else actor.train_dataset
    result.reward_scores, result.reward_metrics = await compute_rewards(actor, result, dataset, is_eval)
    results_queue = actor.eval_results_queue if is_eval else actor.results_queue
    results_queue.put(result)


async def compute_rewards(
    actor: "LLMRayActor", result: GenerationResult, dataset: datasets.Dataset, is_eval: bool
) -> tuple[list[float], dict]:
    index_map = actor._eval_index_map if is_eval else actor._train_index_map
    example = dataset[index_map[result.index]]
    decoded_responses = actor.llm_engine.tokenizer.batch_decode(result.responses, skip_special_tokens=True)

    k = len(result.responses)
    k_ground_truths = [example[GROUND_TRUTHS_KEY]] * k
    k_datasets = [example[VERIFIER_SOURCE_KEY]] * k
    k_raw_queries = [example[RAW_PROMPT_KEY]] * k

    scores, metrics = await actor.reward_fn(
        result.responses,
        decoded_responses,
        k_ground_truths,
        k_datasets,
        result.finish_reasons,
        result.request_info,
        k_raw_queries,
    )
    return scores, metrics


class LLMRayActor:
    """Ray actor for LLM generation with optional tool support."""

    def __init__(
        self,
        *args,
        tool_parser_type: str = "legacy",
        tool_definitions: list[dict] | None = None,
        tool_stop_sequences: list[str] | None = None,
        max_steps: int = 5,
        per_turn_max_tokens: int | None = None,
        mask_tool_use: bool = True,
        pools: dict[str, ray.actor.ActorHandle] | None = None,
        bundle_indices: list[int] | None = None,
        prompt_queue: ray_queue.Queue,
        results_queue: ray_queue.Queue,
        eval_results_queue: ray_queue.Queue,
        actor_manager: ray.actor.ActorHandle,
        inflight_updates: bool,
        reward_config: RewardConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs,
    ):
        assert_threaded_actor(self)
        self._tool_definitions = tool_definitions
        self._tool_stop_sequences = tool_stop_sequences
        self.engine_index = int(kwargs.pop("engine_index", -1))
        self.bundle_indices = list(bundle_indices or [])
        self._init_config(
            max_steps,
            per_turn_max_tokens,
            mask_tool_use,
            pools,
            inflight_updates,
            reward_config,
            train_dataset,
            eval_dataset,
        )
        self._init_queues(prompt_queue, results_queue, eval_results_queue, actor_manager)

        noset_visible_devices = kwargs.pop("noset_visible_devices")
        distributed_executor_backend = kwargs.get("distributed_executor_backend")
        self._setup_gpu_visibility(noset_visible_devices, distributed_executor_backend)
        self._setup_and_start_async_engine(args, bundle_indices, kwargs)
        self._init_openai_client()
        self.inference_batch_size = self.get_kv_cache_info()
        self._init_executor()
        # comes after executor as it requires tokenizer access.
        self._init_tool_parser(tool_parser_type)

    def _init_config(
        self,
        max_steps: int,
        per_turn_max_tokens: int | None,
        mask_tool_use: bool,
        pools: dict[str, ray.actor.ActorHandle] | None,
        inflight_updates: bool,
        reward_config: RewardConfig | None,
        train_dataset,
        eval_dataset,
    ) -> None:
        self.max_steps = max_steps
        self.per_turn_max_tokens = per_turn_max_tokens
        self.mask_tool_use = mask_tool_use
        self.pools: dict[str, ray.actor.ActorHandle] = pools or {}
        self.inflight_updates = inflight_updates
        self.request_metadata = {}
        self.active_tasks = {}
        self.request_outputs = {}
        self.reward_config = reward_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._train_index_map: dict[int, int] = (
            {train_dataset[i]["index"]: i for i in range(len(train_dataset))} if train_dataset is not None else {}
        )
        self._eval_index_map: dict[int, int] = (
            {eval_dataset[i]["index"]: i for i in range(len(eval_dataset))} if eval_dataset is not None else {}
        )
        self.reward_fn = reward_config.build() if reward_config else None
        self.tool_parser: ToolParser  # Set in _init_tool_parser

    def _init_queues(self, prompt_queue, results_queue, eval_results_queue, actor_manager) -> None:
        self.completion_queue = queue.Queue()
        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False

    def _init_executor(self) -> None:
        max_workers = NUM_PREFETCH_WORKERS
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._prefetch_future = self.executor.submit(_prefetch_worker, self)
        self._process_future = self.executor.submit(self.process_from_queue)

    def _init_tool_parser(self, tool_parser_type: str) -> None:
        self.tool_parser = create_tool_parser(
            parser_type=tool_parser_type,
            tokenizer=self.llm_engine.tokenizer,
            tool_definitions=self._tool_definitions,
            stop_sequences=self._tool_stop_sequences,
        )

    @staticmethod
    def _parse_visible_devices(raw_value: str | None) -> list[str]:
        if raw_value is None:
            return []
        return [token.strip() for token in raw_value.split(",") if token.strip()]

    def _maybe_pin_vllm_visible_devices(self) -> bool:
        explicit = (
            os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
            or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
            or ""
        ).strip()
        if explicit:
            os.environ["CUDA_VISIBLE_DEVICES"] = explicit
            logger.info(
                "Pinned vLLM actor CUDA_VISIBLE_DEVICES=%s from explicit env",
                explicit,
            )
            return True

        visible = self._parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not visible:
            return False

        num_engines_raw = os.environ.get("RLVR_VLLM_NUM_ENGINES") or ""
        tensor_parallel_raw = os.environ.get("RLVR_VLLM_TENSOR_PARALLEL") or ""
        trainer_gpus_raw = os.environ.get("RLVR_TRAINING_GPUS") or ""
        try:
            num_engines = int(num_engines_raw) if num_engines_raw else 0
            tensor_parallel = int(tensor_parallel_raw) if tensor_parallel_raw else 0
            trainer_gpus = int(trainer_gpus_raw) if trainer_gpus_raw else 0
        except ValueError:
            return False

        # Auto-pin only for the common single-engine TP run to avoid overlap
        # with trainer GPUs when Ray NOSET is enabled.
        if num_engines != 1 or tensor_parallel <= 1 or trainer_gpus <= 0:
            return False

        start = trainer_gpus
        end = start + tensor_parallel
        if end > len(visible):
            return False

        pinned = ",".join(visible[start:end])
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        logger.info(
            "Auto-pinned vLLM actor CUDA_VISIBLE_DEVICES=%s (trainer_gpus=%d, tp=%d)",
            pinned,
            trainer_gpus,
            tensor_parallel,
        )
        return True

    def _setup_gpu_visibility(self, noset_visible_devices: bool, distributed_executor_backend: str) -> None:
        explicit = (
            os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
            or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
            or ""
        ).strip()
        if distributed_executor_backend == "ray" and noset_visible_devices and explicit:
            # Leave CUDA_VISIBLE_DEVICES untouched at actor start for Ray workers.
            # We pin per-worker later (in Worker.init_device), which avoids Ray's
            # accelerator-id remap mismatch when explicit ids are non-zero.
            logger.info(
                "Using explicit vLLM visibility hint (%s) without actor-level CUDA_VISIBLE_DEVICES override "
                "because ray+NOSET is enabled.",
                explicit,
            )
            return

        # Always honor explicit vLLM visibility first.
        # This is required when rollout GPUs are intentionally separated
        # from trainer GPUs (for example trainer=0,1 and rollout=2,3).
        if self._maybe_pin_vllm_visible_devices():
            return

        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES at the top-level when
        # using the ray backend only if NOSET is explicitly enabled.
        # When NOSET is disabled we keep Ray-assigned visibility so workers
        # get a consistent, non-overlapping device mapping.
        if distributed_executor_backend == "ray" and noset_visible_devices:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in ray.get_gpu_ids())

    def _setup_and_start_async_engine(self, args, bundle_indices, kwargs) -> None:
        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            logger.debug(f"creating LLM with bundle_indices={bundle_indices}")

        explicit = (
            os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
            or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
            or ""
        ).strip()
        if kwargs.get("distributed_executor_backend") == "ray" and explicit:
            os.environ["RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH"] = "1"
            os.environ["RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH"] = "1"

        if kwargs.get("distributed_executor_backend") == "ray":
            _patch_vllm_ray_executor_explicit_visible_devices()

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        engine_args.disable_log_stats = True
        engine_args.disable_cascade_attn = True

        init_complete = threading.Event()
        self.loop = None
        self.llm_engine = None
        self.client = None
        self.server_port = None

        async def _init_engine_and_server():
            running_loop = asyncio.get_running_loop()
            assert running_loop == self.loop, f"Loop mismatch! running={running_loop}, actor.loop={self.loop}"

            engine_client = vllm.AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)

            tokenizer = engine_client.tokenizer
            inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            has_chat_template = getattr(inner_tokenizer, "chat_template", None) is not None
            args = _create_server_args(engine_client.vllm_config.model_config.model, has_chat_template)
            app = build_app(args)
            await init_app_state(engine_client, app.state, args)

            # Create a socket and bind to port 0 to let the OS assign an available port.
            # We pass the socket to serve_http to avoid race conditions where another
            # process could claim the port between bind() and server startup.
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            self.server_port = sock.getsockname()[1]

            logger.info(f"Starting vLLM OpenAI API server on port {self.server_port}")

            config = uvicorn.Config(app, host="127.0.0.1", port=self.server_port, log_level="warning")
            asyncio.create_task(uvicorn.Server(config).serve(sockets=[sock]))

            # Yield control to allow the server task to start before returning.
            await asyncio.sleep(0.1)

            return engine_client

        def _run_loop():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.llm_engine = self.loop.run_until_complete(_init_engine_and_server())
            finally:
                # Signal completion to the waiting main thread even if init failed.
                init_complete.set()
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=_run_loop, daemon=True)
        self.loop_thread.start()

        if init_complete.wait(timeout=INFERENCE_INIT_TIMEOUT_S):
            if self.llm_engine is None:
                raise RuntimeError("vLLM engine initialization failed. Check Ray worker logs for details.")
            return
        message = "timed out" if self.loop_thread.is_alive() else "thread died before completing"
        raise RuntimeError(f"vLLM engine {message}")

    def _init_openai_client(self) -> None:
        base_url = f"http://127.0.0.1:{self.server_port}/v1"
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)
        self.model_name = self.llm_engine.vllm_config.model_config.model

        logger.info(f"Waiting for vLLM OpenAI API server to be ready at {base_url}")

        asyncio.run(_check_health(self.server_port))
        logger.info("vLLM OpenAI API server is ready")

    def get_model_dims(self):
        return model_dims_from_vllm_config(self.llm_engine.vllm_config)

    def get_placement_info(self) -> dict[str, Any]:
        """Return rollout actor placement details for startup validation."""
        node_ip = ""
        try:
            node_ip = str(ray.util.get_node_ip_address())
        except Exception:
            logger.exception("Failed to resolve node IP for rollout actor placement info.")

        node_id = ""
        try:
            node_id = str(ray.get_runtime_context().get_node_id())
        except Exception:
            logger.exception("Failed to resolve Ray node ID for rollout actor placement info.")

        gpu_ids: list[str] = []
        try:
            gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()]
        except Exception:
            logger.exception("Failed to resolve Ray GPU IDs for rollout actor placement info.")

        tensor_parallel_size = 0
        try:
            tensor_parallel_size = int(
                self.llm_engine.vllm_config.parallel_config.tensor_parallel_size,
            )
        except Exception:
            logger.exception(
                "Failed to resolve tensor_parallel_size for rollout actor placement info.",
            )

        return {
            "engine_index": self.engine_index,
            "node_ip": node_ip,
            "ray_node_id": node_id,
            "gpu_ids": gpu_ids,
            "bundle_indices": list(self.bundle_indices),
            "tensor_parallel_size": tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "rlvr_vllm_cuda_visible_devices": os.environ.get(
                "RLVR_VLLM_CUDA_VISIBLE_DEVICES",
                "",
            ),
        }

    def _should_stop(self) -> bool:
        if self.actor_manager is None:
            return self._should_stop_value
        if (time.perf_counter() - self._last_should_stop_update) > SHOULD_STOP_TIMEOUT_S:
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=SHOULD_STOP_TIMEOUT_S)
            if ready_refs:
                self._should_stop_value = ray.get(ready_refs[0])
                self._last_should_stop_update = time.perf_counter()
            else:
                ray.cancel(should_stop_ref)
        return self._should_stop_value

    def process_from_queue(self) -> None:
        finalize_futures: list[futures.Future] = []
        while True:
            completion_future = accumulate_completions(self, self.completion_queue.get())
            if completion_future is not None:
                finalize_futures.append(completion_future)

            done, not_done = futures.wait(finalize_futures, timeout=0)
            [future.result() for future in done]
            finalize_futures = list(not_done)

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        use_ray: bool = False,
        timeout_minutes: int = 120,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.collective_rpc(
                "init_process_group",
                args=(
                    master_address,
                    master_port,
                    rank_offset,
                    world_size,
                    group_name,
                    backend,
                    use_ray,
                    timeout_minutes,
                ),
            ),
            self.loop,
        )
        return future.result(timeout=timeout_minutes * 60)

    def _run_async(self, coro: Awaitable[Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def _prepare_weight_update(self, name: str, dtype: str) -> None:
        # Wait for all active requests to complete.
        while not self.inflight_updates and len(self.active_tasks) > 0:
            self.check_background_threads()
            time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)

        expected_dtype = str(self.llm_engine.model_config.dtype)
        assert dtype == expected_dtype, f"Mismatched dtype for {name}: received {dtype!r}, expected {expected_dtype!r}"

    def update_weight(self, name: str, dtype: str, shape: tuple[int, ...], empty_cache: bool = False) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache)))

    def update_weight_cuda_ipc(
        self, name: str, dtype: str, shape: tuple[int, ...], ipc_handles: list[Any], empty_cache: bool = False
    ) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(
            self.llm_engine.collective_rpc(
                "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
            )
        )

    def reset_prefix_cache(self) -> None:
        return self._run_async(self.llm_engine.reset_prefix_cache())

    def ready(self) -> bool:
        return True

    def check_background_threads(self) -> None:
        if self._prefetch_future.done():
            self._prefetch_future.result()
        if self._process_future.done():
            self._process_future.result()
        for task in self.active_tasks.values():
            if task.done():
                task.result()
        if not self.loop_thread.is_alive():
            raise RuntimeError(
                "vLLM engine loop thread has died. Check logs for errors in EngineCore or async engine."
            )

    def get_kv_cache_info(self) -> int:
        """Get KV cache max concurrency from the vLLM engine."""
        kv_cache_specs = self._run_async(self.llm_engine.collective_rpc("get_kv_cache_spec"))

        vllm_config = self.llm_engine.vllm_config
        gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory_utilization * total_gpu_memory)

        kv_cache_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs[0])

        kv_cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

        return int(max_concurrency)


async def process_request(actor: LLMRayActor, sub_request_id: str, sampling_params: SamplingConfig):
    """Process a single async request with tool/environment support."""
    base_request_id = split_request_id(sub_request_id)["base_id"]
    request_metadata = actor.request_metadata.get(base_request_id)

    response_tokens: list[int] = []
    response_logprobs: list[float] = []
    response_masks: list[int] = []
    cumulative_logprob = 0.0
    rollout = RolloutState()

    acquired: dict[str, tuple[Any, Any]] = {}
    actor_map: dict[str, Any] = {}
    env_config: dict[str, Any] | None = None
    env_name: str | None = None
    env_response_role = "tool"
    is_text_env = False
    output = None
    request_error: Exception | None = None

    try:
        # Per-request health checks are expensive and can become a failure hotspot
        # under high-concurrency rollout traffic. Keep them opt-in via env.
        if REQUEST_HEALTH_CHECK_ENABLED:
            await _check_health(actor.server_port)

        if request_metadata is None:
            raise KeyError(f"Missing request metadata for {base_request_id}")

        original_prompt = request_metadata["prompt_token_ids"]
        active_tools = request_metadata["active_tools"]
        env_config = request_metadata.get("env_config")
        current_prompt = list(original_prompt)
        max_model_len = actor.llm_engine.model_config.max_model_len

        configured_tools = set(actor.pools.keys())
        allowed_tools = configured_tools & set(active_tools) if active_tools is not None else configured_tools
        max_steps = env_config.get("max_steps", actor.max_steps) if env_config else actor.max_steps

        if env_config is not None:
            env_name = env_config["env_name"]
            pool = actor.pools.get(env_name)
            if pool is None:
                raise ValueError(f"No pool for env '{env_name}'. Available: {list(actor.pools.keys())}")
            env_actor = await pool.acquire.remote()
            acquired[env_name] = (pool, env_actor)
            env_kwargs = {
                k: v for k, v in env_config.items() if k not in ("env_name", "max_steps", "pool_size", "is_text_env")
            }
            _, env_tools = await env_actor.reset.remote(**env_kwargs)
            env_response_role = await env_actor.get_response_role.remote()
            is_text_env = env_config.get("is_text_env", False)

            if env_tools:
                env_tool_names = {t["function"]["name"] for t in env_tools}
                clashes = env_tool_names & set(actor.pools.keys())
                if clashes:
                    raise ValueError(
                        f"Env '{env_name}' tool names clash with tool pool names: {sorted(clashes)}. "
                        f"Rename one side to avoid ambiguous dispatch."
                    )
                for name in env_tool_names:
                    actor_map[name] = env_actor
                    allowed_tools.add(name)

            if is_text_env:
                actor_map[env_name] = env_actor
                allowed_tools.add(env_name)

        while rollout.step_count < max_steps:
            if rollout.done:
                break

            remaining_budget = sampling_params.max_tokens - len(response_masks)
            remaining_room = max_model_len - len(current_prompt)
            if remaining_budget <= 0 or remaining_room <= 0:
                break

            per_turn_budget = actor.per_turn_max_tokens if actor.per_turn_max_tokens is not None else remaining_budget
            current_max_tokens = max(1, min(remaining_budget, remaining_room, per_turn_budget))
            current_sampling_params = dataclasses.replace(sampling_params, max_tokens=current_max_tokens)
            api_response = await actor.client.completions.create(
                model=actor.model_name,
                prompt=current_prompt,
                extra_body={
                    "return_token_ids": True,
                    "cache_salt": base_request_id,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                },
                **dataclasses.asdict(current_sampling_params),
            )

            output = api_response.choices[0]
            model_tokens = list(output.token_ids)
            response_tokens.extend(model_tokens)
            current_prompt.extend(model_tokens)

            assert output.logprobs and output.logprobs.token_logprobs, "logprobs must be available"
            for logprob in output.logprobs.token_logprobs:
                response_logprobs.append(logprob)
                cumulative_logprob += logprob
            response_masks.extend([1] * len(model_tokens))

            tool_calls = [tc for tc in actor.tool_parser.get_tool_calls(output.text) if tc.name in allowed_tools]

            # Text envs: inject a shadow tool call so dispatch handles it uniformly
            if is_text_env:
                tool_calls.append(EnvCall(id="", name=env_name, args={"text": output.text}))

            if not tool_calls:
                break

            observations: list[str] = []
            for tc in tool_calls:
                if rollout.step_count >= max_steps:
                    break

                # Lazily acquire from pool on first use of this tool name
                if tc.name not in actor_map:
                    pool = actor.pools.get(tc.name)
                    if pool is None:
                        raise ValueError(
                            f"Model called tool '{tc.name}' but no pool exists for it. "
                            f"Available pools: {list(actor.pools.keys())}"
                        )
                    acq = await pool.acquire.remote()
                    acquired[tc.name] = (pool, acq)
                    actor_map[tc.name] = acq
                target = actor_map[tc.name]

                rollout.step_count += 1

                try:
                    step_result: StepResult = await target.step.remote(
                        EnvCall(id=str(rollout.step_count), name=tc.name, args=tc.args)
                    )
                    observations.append(step_result.result)
                    rollout.tool_output += step_result.result
                    rollout.rewards.append(step_result.reward)
                    if step_result.done:
                        rollout.done = True
                    meta = step_result.metadata or {}
                    rollout.timeout = rollout.timeout or meta.get("timeout", False)
                    rollout.tool_error += meta.get("error", "")
                    rollout.tool_runtime += meta.get("runtime", 0.0)
                    rollout.tool_call_stats.append(
                        ToolCallStats(
                            tool_name=tc.name,
                            success=not meta.get("error") and not meta.get("timeout", False),
                            runtime=meta.get("runtime", 0.0),
                        )
                    )
                except Exception as e:
                    error_msg = f"Step '{tc.name}' failed: {e}. Args: {tc.args}"
                    logger.warning(error_msg)
                    observations.append(error_msg)
                    rollout.tool_error += error_msg
                    rollout.rewards.append(0.0)
                    rollout.tool_call_stats.append(ToolCallStats(tool_name=tc.name, success=False, runtime=0.0))

                if rollout.done:
                    break

            if observations:
                tokens, logprobs, masks, excess = process_tool_tokens(
                    observations,
                    actor.tool_parser,
                    actor.llm_engine.tokenizer,
                    len(current_prompt),
                    len(response_masks),
                    max_model_len,
                    sampling_params.max_tokens,
                    actor.mask_tool_use,
                    role=env_response_role,
                )
                response_tokens.extend(tokens)
                response_logprobs.extend(logprobs)
                response_masks.extend(masks)
                current_prompt.extend(tokens)
                if excess > 0:
                    break
    except Exception as exc:  # noqa: BLE001
        request_error = exc
        if isinstance(exc, TimeoutError):
            rollout.timeout = True
        error_text = f"{type(exc).__name__}: {exc}"
        rollout.tool_error = f"{rollout.tool_error}\n{error_text}".strip()
    finally:
        if env_config is not None and env_name is not None and env_name in acquired:
            _, env_act = acquired[env_name]
            rollout.info["env_name"] = env_name
            rollout.info.update(await env_act.get_metrics.remote())
        for pool, acq_actor in acquired.values():
            pool.release.remote(acq_actor)
        actor.active_tasks.pop(sub_request_id, None)

    if request_error is not None:
        if isinstance(request_error, TimeoutError):
            logger.warning(
                "Request %s failed in process_request (%s): %s. Returning fallback completion.",
                sub_request_id,
                type(request_error).__name__,
                request_error,
            )
        else:
            logger.warning(
                "Request %s failed in process_request (%s): %s. Returning fallback completion.",
                sub_request_id,
                type(request_error).__name__,
                request_error,
                exc_info=(type(request_error), request_error, request_error.__traceback__),
            )

    if len(response_tokens) == 0:
        eos_token_id = actor.llm_engine.tokenizer.eos_token_id
        response_tokens = [eos_token_id] if eos_token_id is not None else []
        response_logprobs = [float("nan")] if response_tokens else []
        response_masks = [1] * len(response_tokens)

    finish_reason = output.finish_reason if output else "stop"
    complete_output = CompletionOutput(
        index=split_request_id(sub_request_id)["request_index"],
        token_ids=response_tokens,
        cumulative_logprob=cumulative_logprob,
        logprobs=response_logprobs,
        finish_reason=finish_reason,
        mask=response_masks,
        rollout_state=dataclasses.asdict(rollout),
    )

    metadata = actor.request_metadata.get(base_request_id)
    if metadata is None:
        logger.warning(
            "Dropping completion for request %s because metadata for %s is missing (likely during shutdown).",
            sub_request_id,
            base_request_id,
        )
        return

    actor.completion_queue.put(
        {
            "base_request_id": base_request_id,
            "expected_n": metadata["original_sampling_params"].n,
            "request_output": RequestOutput(
                request_id=sub_request_id,
                prompt_token_ids=metadata["prompt_token_ids"],
                outputs=[complete_output],
            ),
            "use_tools": bool(actor.pools),
        }
    )


def get_cuda_arch_list() -> str:
    """Get CUDA compute capabilities and format them for TORCH_CUDA_ARCH_LIST."""
    if not torch.cuda.is_available():
        return ""

    cuda_capabilities = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_capabilities.append(f"{major}.{minor}")

    # Remove duplicates and sort
    cuda_capabilities = sorted(set(cuda_capabilities))
    cuda_arch_list = ";".join(cuda_capabilities)
    logger.info(
        f"Detected CUDA compute capabilities: {cuda_capabilities}, setting TORCH_CUDA_ARCH_LIST={cuda_arch_list}"
    )
    return cuda_arch_list


def _normalize_engine_node_ips(
    engine_node_ips: list[str] | None,
    num_engines: int,
    *,
    enforce: bool,
) -> list[str] | None:
    """Normalize and validate explicit per-engine node mapping."""
    if engine_node_ips is None:
        if enforce:
            raise ValueError(
                "enforce_engine_node_ips=True requires engine_node_ips to be provided.",
            )
        return None

    normalized = [node_ip.strip() for node_ip in engine_node_ips]
    if any(not node_ip for node_ip in normalized):
        raise ValueError("engine_node_ips must not contain empty node IP values.")
    if len(normalized) != num_engines:
        raise ValueError(
            "engine_node_ips length must equal num_engines. "
            f"Got {len(normalized)} for num_engines={num_engines}.",
        )
    return normalized


def _normalize_engine_visible_devices(
    engine_visible_devices: list[str] | None,
    num_engines: int,
    *,
    tensor_parallel_size: int,
) -> list[str] | None:
    """Normalize and validate optional per-engine explicit CUDA visibility."""
    if engine_visible_devices is None:
        return None

    normalized: list[str] = []
    for raw_value in engine_visible_devices:
        visible_tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        if len(visible_tokens) != tensor_parallel_size:
            raise ValueError(
                "Each engine_visible_devices entry must contain exactly "
                f"tensor_parallel_size={tensor_parallel_size} GPU ids. "
                f"Got '{raw_value}' ({len(visible_tokens)} ids)."
            )
        normalized.append(",".join(visible_tokens))

    if len(normalized) != num_engines:
        raise ValueError(
            "engine_visible_devices length must equal num_engines. "
            f"Got {len(normalized)} for num_engines={num_engines}.",
        )
    return normalized


def _build_rollout_pg_bundles(
    num_engines: int,
    tensor_parallel_size: int,
    engine_node_ips: list[str] | None = None,
) -> tuple[list[dict[str, float]], list[list[int]]]:
    """Build placement-group bundles and per-engine bundle index groups."""
    if num_engines <= 0:
        raise ValueError(f"num_engines must be > 0, got {num_engines}.")
    if tensor_parallel_size <= 0:
        raise ValueError(
            "tensor_parallel_size must be > 0, "
            f"got {tensor_parallel_size}.",
        )
    if engine_node_ips is not None and len(engine_node_ips) != num_engines:
        raise ValueError(
            "engine_node_ips length must equal num_engines. "
            f"Got {len(engine_node_ips)} for num_engines={num_engines}.",
        )

    bundles: list[dict[str, float]] = []
    bundle_indices_by_engine: list[list[int]] = []
    for engine_idx in range(num_engines):
        engine_bundle_indices: list[int] = []
        node_ip = engine_node_ips[engine_idx] if engine_node_ips is not None else None
        for _ in range(tensor_parallel_size):
            bundle: dict[str, float] = {"GPU": 1.0, "CPU": 1.0}
            if node_ip is not None:
                bundle[f"node:{node_ip}"] = 0.01
            bundles.append(bundle)
            engine_bundle_indices.append(len(bundles) - 1)
        bundle_indices_by_engine.append(engine_bundle_indices)
    return bundles, bundle_indices_by_engine


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    tokenizer_name_or_path: str,
    pretrain: str,
    revision: str | None,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    vllm_gpu_memory_utilization: float = 0.9,
    disable_custom_all_reduce: bool = False,
    single_gpu_mode: bool = False,
    pg: PlacementGroup | None = None,
    tool_parser_type: str = "legacy",
    tool_definitions: list[dict] | None = None,
    tool_stop_sequences: list[str] | None = None,
    max_steps: int = 5,
    per_turn_max_tokens: int | None = None,
    mask_tool_use: bool = True,
    pools: dict[str, ray.actor.ActorHandle] | None = None,
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
    inflight_updates: bool = False,
    reward_config: RewardConfig | None = None,
    train_dataset=None,
    eval_dataset=None,
    vllm_dtype: str = "bfloat16",
    engine_node_ips: list[str] | None = None,
    engine_visible_devices: list[str] | None = None,
    enforce_engine_node_ips: bool = False,
) -> list[ray.actor.ActorHandle]:
    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    noset_visible = ray_noset_visible_devices()
    explicit_vllm_visible = (
        os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
        or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()
    normalized_engine_node_ips = _normalize_engine_node_ips(
        engine_node_ips,
        num_engines,
        enforce=enforce_engine_node_ips,
    )
    normalized_engine_visible_devices = _normalize_engine_visible_devices(
        engine_visible_devices,
        num_engines,
        tensor_parallel_size=tensor_parallel_size,
    )
    if explicit_vllm_visible and normalized_engine_visible_devices is not None:
        raise ValueError(
            "engine_visible_devices cannot be combined with global explicit "
            "RLVR_VLLM_CUDA_VISIBLE_DEVICES/VLLM_CUDA_VISIBLE_DEVICES.",
        )
    use_hybrid_engine = pg is not None and not explicit_vllm_visible
    if tensor_parallel_size != 1 and use_hybrid_engine:
        raise ValueError("tensor_parallel_size > 1 is not supported with single_gpu_mode")
    if pg is not None and explicit_vllm_visible:
        logger.info(
            "Disabling hybrid vLLM placement because explicit visibility is set "
            "(RLVR_VLLM_CUDA_VISIBLE_DEVICES/VLLM_CUDA_VISIBLE_DEVICES=%s).",
            explicit_vllm_visible,
        )
    if use_hybrid_engine and normalized_engine_node_ips is not None:
        msg = (
            "engine_node_ips is not supported when using hybrid vLLM placement "
            "(single_gpu_mode)."
        )
        if enforce_engine_node_ips:
            raise ValueError(msg)
        logger.warning("%s Ignoring explicit engine_node_ips.", msg)
        normalized_engine_node_ips = None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1 and single_gpu_mode:
        # every worker will use 0.5 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.5

    logger.info(f"num_gpus: {num_gpus}")

    explicit_bundle_indices_by_engine: list[list[int]] = []
    if not use_hybrid_engine:
        bundles, explicit_bundle_indices_by_engine = _build_rollout_pg_bundles(
            num_engines,
            tensor_parallel_size,
            normalized_engine_node_ips,
        )
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    runtime_env_vars = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TORCH_CUDA_ARCH_LIST": get_cuda_arch_list(),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    }
    if distributed_executor_backend == "ray":
        # Enable scoped RLVR sitecustomize patches for all rollout Ray actors.
        # This keeps nested vLLM Ray worker accelerator-id handling stable even
        # when explicit rollout visibility is not configured.
        runtime_env_vars.update(
            {
                "RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH": "1",
                "RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH": "1",
            }
        )
    if explicit_vllm_visible:
        # Ensure nested vLLM Ray workers inherit the intended rollout GPU slice.
        runtime_env_vars.update(
            {
                "RLVR_VLLM_CUDA_VISIBLE_DEVICES": explicit_vllm_visible,
                "VLLM_CUDA_VISIBLE_DEVICES": explicit_vllm_visible,
                "RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH": "1",
                "RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH": "1",
            }
        )

    if normalized_engine_node_ips is not None and not use_hybrid_engine:
        # Preserve engine-to-bundle order in explicit placement mode.
        bundle_indices_by_engine = explicit_bundle_indices_by_engine
    else:
        # Legacy behavior: use Ray placement order to maximize node-local TP grouping.
        bundle_indices_list = get_bundle_indices_list(pg)
        bundle_indices_by_engine = [
            bundle_indices_list[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]
            for i in range(num_engines)
        ]

    for i in range(num_engines):
        bundle_indices = bundle_indices_by_engine[i]
        if len(bundle_indices) != tensor_parallel_size:
            raise RuntimeError(
                "Resolved bundle count does not match tensor parallel size for "
                f"engine {i}: expected {tensor_parallel_size}, got {len(bundle_indices)}.",
            )

        runtime_env_vars_for_engine = dict(runtime_env_vars)
        engine_explicit_visible = (
            normalized_engine_visible_devices[i]
            if normalized_engine_visible_devices is not None
            else None
        )
        if engine_explicit_visible:
            runtime_env_vars_for_engine.update(
                {
                    "RLVR_VLLM_CUDA_VISIBLE_DEVICES": engine_explicit_visible,
                    "VLLM_CUDA_VISIBLE_DEVICES": engine_explicit_visible,
                    "RLVR_APPLY_VLLM_RAY_VISIBLE_PATCH": "1",
                    "RLVR_ENABLE_VLLM_SITECUSTOMIZE_PATCH": "1",
                }
            )

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0],
        )

        if normalized_engine_node_ips is not None:
            logger.info(
                "Creating rollout engine %d on node=%s bundles=%s explicit_visible=%s",
                i,
                normalized_engine_node_ips[i],
                bundle_indices,
                engine_explicit_visible or "<auto>",
            )
        else:
            logger.info(
                "Creating rollout engine %d bundles=%s explicit_visible=%s",
                i,
                bundle_indices,
                engine_explicit_visible or "<auto>",
            )

        vllm_engines.append(
            ray.remote(LLMRayActor)
            .options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars=runtime_env_vars_for_engine
                ),
            )
            .remote(
                model=pretrain,
                revision=revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=revision,
                worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                disable_custom_all_reduce=disable_custom_all_reduce,
                dtype=vllm_dtype,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                noset_visible_devices=noset_visible,
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                actor_manager=actor_manager,
                tool_parser_type=tool_parser_type,
                tool_definitions=tool_definitions,
                tool_stop_sequences=tool_stop_sequences,
                max_steps=max_steps,
                per_turn_max_tokens=per_turn_max_tokens,
                mask_tool_use=mask_tool_use,
                pools=pools,
                inflight_updates=inflight_updates,
                reward_config=reward_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                engine_index=i,
            )
        )

    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=1200
    )

    return vllm_engines


def _send_to_vllm(
    name: str,
    param: torch.nn.Parameter,
    is_last: bool,
    deepspeed_stage: int,
    vllm_engines: list[ray.actor.ActorHandle],
    model_update_group: torch.distributed.ProcessGroup,
) -> list[ray.ObjectRef]:
    """Send a parameter to vLLM engines via broadcast."""
    shape = param.ds_shape if deepspeed_stage == 3 else param.shape
    refs = [
        engine.update_weight.remote(name, dtype=str(param.dtype), shape=shape, empty_cache=is_last)
        for engine in vllm_engines
    ]
    torch.distributed.broadcast(param.data, 0, group=model_update_group)
    return refs


def broadcast_weights_to_vllm(
    model: torch.nn.Module,
    vllm_engines: list[ray.actor.ActorHandle],
    model_update_group: torch.distributed.ProcessGroup | None,
    deepspeed_stage: int,
    gather_whole_model: bool = True,
) -> list[ray.ObjectRef]:
    """Broadcast DeepSpeed model weights to vLLM engines.

    Must be called on ALL ranks when using DeepSpeed stage 3, since
    GatheredParameters is a collective operation. Only rank 0 actually
    sends weights to vLLM.

    Args:
        model: The unwrapped model (model.module from DeepSpeed engine)
        vllm_engines: List of vLLM engine actor handles
        model_update_group: Process group for distributed broadcast (only needed on rank 0)
        deepspeed_stage: DeepSpeed ZeRO stage (3 requires GatheredParameters)
        gather_whole_model: If True, gather all params at once (more memory, faster).
            If False, gather each param individually (less memory, slower).

    Returns:
        List of Ray ObjectRefs for the weight update calls (empty on non-rank-0)
    """
    is_rank_0 = torch.distributed.get_rank() == 0
    params = list(model.named_parameters())
    num_params = len(params)
    all_refs: list[ray.ObjectRef] = []

    if gather_whole_model:
        with deepspeed.zero.GatheredParameters(model.parameters(), enabled=deepspeed_stage == 3):
            if is_rank_0:
                for i, (name, param) in enumerate(params):
                    all_refs.extend(
                        _send_to_vllm(
                            name, param, i == num_params - 1, deepspeed_stage, vllm_engines, model_update_group
                        )
                    )
    else:
        for i, (name, param) in enumerate(params):
            with deepspeed.zero.GatheredParameters([param], enabled=deepspeed_stage == 3):
                if is_rank_0:
                    all_refs.extend(
                        _send_to_vllm(
                            name, param, i == num_params - 1, deepspeed_stage, vllm_engines, model_update_group
                        )
                    )

    return all_refs
