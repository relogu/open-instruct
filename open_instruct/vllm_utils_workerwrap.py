import os


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _debug_enabled() -> bool:
    return _is_truthy(os.environ.get("RLVR_DEBUG_SITECUSTOMIZE"))


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[rlvr-workerwrap pid={os.getpid()}] {message}", flush=True)


def _explicit_rollout_visible_devices() -> str:
    return (
        os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES")
        or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()


def _parse_visible_ids(raw_value: str | None) -> list[int]:
    if not raw_value:
        return []
    ids: list[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if token.isdigit():
            ids.append(int(token))
    return ids


def _resolve_ray_assigned_cuda_index() -> int | None:
    """Resolve CUDA index from Ray-assigned GPU id and current visibility."""
    try:
        import ray
    except Exception:
        return None

    try:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    except Exception:
        return None

    gpu_ids = accelerator_ids.get("GPU") or accelerator_ids.get("gpu") or []
    if not gpu_ids:
        return None

    selected: int | None = None
    for gpu_id in gpu_ids:
        try:
            selected = int(float(gpu_id))
            break
        except Exception:
            continue
    if selected is None:
        return None

    current_visible = _parse_visible_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if current_visible:
        if selected in current_visible:
            return current_visible.index(selected)
        if 0 <= selected < len(current_visible):
            return selected
        return None
    return selected


def _apply_explicit_vllm_cuda_visible_devices() -> tuple[str | None, int]:
    explicit = _explicit_rollout_visible_devices()
    if not explicit:
        return None, 0
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if (
        current_visible
        and current_visible != explicit
        and not os.environ.get("RLVR_ORIGINAL_CUDA_VISIBLE_DEVICES")
    ):
        os.environ["RLVR_ORIGINAL_CUDA_VISIBLE_DEVICES"] = current_visible
    os.environ["CUDA_VISIBLE_DEVICES"] = explicit
    return explicit, len(_parse_visible_ids(explicit))


def _patch_ray_visible_accelerator_env_hooks() -> None:
    """Keep explicit rollout visibility stable across actor method boundaries.

    Ray wraps actor method calls with a set/reset sequence on accelerator
    visibility env vars. In worker wrappers we force this to no-op when
    explicit rollout visibility is requested.
    """
    if not _explicit_rollout_visible_devices():
        return

    try:
        import ray._private.utils as ray_private_utils
    except Exception:
        return

    if not getattr(ray_private_utils, "_rlvr_workerwrap_set_visible_patch_applied", False):
        original_set_visible = getattr(ray_private_utils, "set_visible_accelerator_ids", None)
        if original_set_visible is not None:
            def _patched_set_visible_accelerator_ids():
                if _is_truthy(os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")) and _explicit_rollout_visible_devices():
                    return {}
                return original_set_visible()

            ray_private_utils.set_visible_accelerator_ids = _patched_set_visible_accelerator_ids
            ray_private_utils._rlvr_workerwrap_set_visible_patch_applied = True
            _debug("Patched Ray set_visible_accelerator_ids in worker wrapper process")

    if not getattr(ray_private_utils, "_rlvr_workerwrap_reset_visible_patch_applied", False):
        original_reset_visible = getattr(ray_private_utils, "reset_visible_accelerator_env_vars", None)
        if original_reset_visible is not None:
            def _patched_reset_visible_accelerator_env_vars(original_visible_accelerator_env_vars):
                explicit = _explicit_rollout_visible_devices()
                if _is_truthy(os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")) and explicit:
                    if isinstance(original_visible_accelerator_env_vars, dict):
                        filtered = dict(original_visible_accelerator_env_vars)
                        filtered.pop("CUDA_VISIBLE_DEVICES", None)
                        return original_reset_visible(filtered)
                return original_reset_visible(original_visible_accelerator_env_vars)

            ray_private_utils.reset_visible_accelerator_env_vars = _patched_reset_visible_accelerator_env_vars
            ray_private_utils._rlvr_workerwrap_reset_visible_patch_applied = True
            _debug("Patched Ray reset_visible_accelerator_env_vars in worker wrapper process")


def _patch_ray_worker_accelerator_id_resolution() -> None:
    """Handle explicit CUDA visibility with Ray GPU ids from physical ordinals.

    When worker processes start with explicit CUDA_VISIBLE_DEVICES such as "2,3",
    Ray may still report assigned GPU resource ids as physical ordinals (2/3).
    Older Ray mapping logic indexes into original_visible_accelerator_ids using
    those ordinals directly, which can raise IndexError for short explicit lists.
    """
    if not _explicit_rollout_visible_devices():
        return

    try:
        import ray._private.worker as ray_private_worker
    except Exception:
        return

    worker_cls = getattr(ray_private_worker, "Worker", None)
    if worker_cls is None:
        return

    original_get = getattr(worker_cls, "get_accelerator_ids_for_accelerator_resource", None)
    if original_get is None:
        return
    if getattr(original_get, "_rlvr_cuda_patch", False):
        return

    def _patched_get(self, resource_name, resource_regex):
        try:
            return original_get(self, resource_name, resource_regex)
        except IndexError:
            resource_ids = self.core_worker.resource_ids()
            assigned_ids: set[int | str] = set()
            import re

            for resource, assignment in resource_ids.items():
                if resource == resource_name or re.match(resource_regex, resource):
                    for resource_id, _ in assignment:
                        assigned_ids.add(resource_id)

            original_ids = self.original_visible_accelerator_ids.get(resource_name, None)
            if original_ids is None:
                return list(assigned_ids)

            resolved: set[str] = set()
            for resource_id in assigned_ids:
                try:
                    idx = int(resource_id)
                except Exception:
                    resolved.add(str(resource_id))
                    continue

                if 0 <= idx < len(original_ids):
                    resolved.add(str(original_ids[idx]))
                else:
                    resolved.add(str(idx))

            _debug(
                "Patched Ray accelerator id fallback "
                f"resource={resource_name} assigned={sorted(str(x) for x in assigned_ids)} "
                f"original_visible={list(original_ids)} resolved={sorted(resolved)}"
            )
            return list(resolved)

    _patched_get._rlvr_cuda_patch = True
    worker_cls.get_accelerator_ids_for_accelerator_resource = _patched_get
    _debug("Patched Ray Worker.get_accelerator_ids_for_accelerator_resource")


def _patch_worker_wrapper_execute_method() -> None:
    """Re-apply explicit rollout visibility before each wrapper method call."""
    try:
        from vllm.worker.worker_base import WorkerWrapperBase
    except Exception:
        return

    original_execute_method = getattr(WorkerWrapperBase, "execute_method", None)
    if original_execute_method is None:
        return
    if getattr(original_execute_method, "_rlvr_cuda_patch", False):
        return

    def _patched_execute_method(self, method, *args, **kwargs):
        explicit, _ = _apply_explicit_vllm_cuda_visible_devices()
        if explicit and method in {"update_environment_variables", "init_worker", "init_device"}:
            _debug(
                "execute_method="
                f"{method} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"RLVR_VLLM_CUDA_VISIBLE_DEVICES={os.environ.get('RLVR_VLLM_CUDA_VISIBLE_DEVICES')} "
                f"VLLM_CUDA_VISIBLE_DEVICES={os.environ.get('VLLM_CUDA_VISIBLE_DEVICES')}"
            )
        return original_execute_method(self, method, *args, **kwargs)

    _patched_execute_method._rlvr_cuda_patch = True
    WorkerWrapperBase.execute_method = _patched_execute_method


def _patch_vllm_worker_init_device() -> None:
    """Patch vLLM worker init_device so explicit visibility is applied early."""
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:
        return

    original_init_device = getattr(Worker, "init_device", None)
    if original_init_device is None:
        return
    if getattr(original_init_device, "_rlvr_cuda_patch", False):
        return

    def _patched_init_device(self, *args, **kwargs):
        explicit, num_visible = _apply_explicit_vllm_cuda_visible_devices()
        mapped_index = None
        if num_visible > 0:
            explicit_ids = _parse_visible_ids(explicit)
            worker_index = None
            for attr in ("rank", "global_rank", "worker_rank", "local_rank"):
                value = getattr(self, attr, None)
                if isinstance(value, int):
                    worker_index = value
                    break
                if isinstance(value, str) and value.isdigit():
                    worker_index = int(value)
                    break
            if worker_index is None:
                env_rank = os.environ.get("RANK")
                if env_rank and env_rank.isdigit():
                    worker_index = int(env_rank)
            if worker_index is None:
                worker_index = 0

            # Primary path: derive a deterministic per-engine offset from the
            # placement-group bundle indices assigned to this actor, then place
            # each TP worker within that actor onto consecutive explicit GPUs.
            # Example:
            #   explicit_ids = [2,3,4,5]
            #   actor A bundle_indices = [0,1] -> mapped_index 0,1 (GPU 2,3)
            #   actor B bundle_indices = [2,3] -> mapped_index 2,3 (GPU 4,5)
            if mapped_index is None and explicit_ids:
                bundle_indices_raw = os.environ.get("VLLM_RAY_BUNDLE_INDICES", "")
                bundle_indices: list[int] = []
                if bundle_indices_raw:
                    for token in bundle_indices_raw.split(","):
                        token = token.strip()
                        if token.isdigit():
                            bundle_indices.append(int(token))
                if bundle_indices:
                    tp_span = len(bundle_indices)
                    base_slot = min(bundle_indices)
                    local_tp_rank = worker_index % tp_span
                    mapped_index = (base_slot + local_tp_rank) % num_visible

            # Prefer the original per-worker CUDA visibility captured before
            # overriding CUDA_VISIBLE_DEVICES with the explicit rollout list.
            # This keeps Ray's worker->GPU assignment stable.
            if mapped_index is None:
                original_visible = _parse_visible_ids(
                    os.environ.get("RLVR_ORIGINAL_CUDA_VISIBLE_DEVICES")
                )
                if original_visible:
                    original_physical = original_visible[worker_index % len(original_visible)]
                    if original_physical in explicit_ids:
                        mapped_index = explicit_ids.index(original_physical)
                    elif not explicit_ids and 0 <= original_physical < num_visible:
                        mapped_index = original_physical

            # Fall back to Ray's accelerator ids when available.
            if mapped_index is None:
                try:
                    import ray

                    ctx = ray.get_runtime_context()
                    accelerator_ids = ctx.get_accelerator_ids()
                    gpu_ids = accelerator_ids.get("GPU") or accelerator_ids.get("gpu") or []
                    if gpu_ids:
                        normalized_gpu_ids: list[int] = []
                        for gpu_id in gpu_ids:
                            try:
                                normalized_gpu_ids.append(int(float(gpu_id)))
                            except Exception:
                                pass
                        if normalized_gpu_ids:
                            selected_id = normalized_gpu_ids[worker_index % len(normalized_gpu_ids)]
                            if explicit_ids and selected_id in explicit_ids:
                                mapped_index = explicit_ids.index(selected_id)
                            elif not explicit_ids and 0 <= selected_id < num_visible:
                                mapped_index = selected_id
                except Exception:
                    pass

            # Final fallback: a deterministic index from inferred worker rank.
            if mapped_index is None:
                mapped_index = worker_index % num_visible

            if mapped_index is None:
                mapped_index = 0
        else:
            # When explicit rollout visibility is not set, prefer Ray's assigned
            # GPU id over rank-based defaults to avoid colliding with policy GPUs.
            mapped_index = _resolve_ray_assigned_cuda_index()

        if mapped_index is not None:
            self.local_rank = mapped_index
        if explicit:
            _debug(
                "worker.init_device "
                f"local_rank={getattr(self, 'local_rank', None)} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"RLVR_VLLM_CUDA_VISIBLE_DEVICES={os.environ.get('RLVR_VLLM_CUDA_VISIBLE_DEVICES')} "
                f"VLLM_CUDA_VISIBLE_DEVICES={os.environ.get('VLLM_CUDA_VISIBLE_DEVICES')}"
            )
        return original_init_device(self, *args, **kwargs)

    _patched_init_device._rlvr_cuda_patch = True
    Worker.init_device = _patched_init_device


_patch_ray_visible_accelerator_env_hooks()
_patch_ray_worker_accelerator_id_resolution()
_patch_worker_wrapper_execute_method()
_patch_vllm_worker_init_device()


class WorkerWrap:
    @staticmethod
    def _parse_visible_ids(raw_value: str | None) -> list[int]:
        return _parse_visible_ids(raw_value)

    def _resolve_cuda_index(self, torch) -> int:
        """Resolve a stable CUDA device index for this worker process."""
        if not torch.cuda.is_available():
            return 0

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return 0

        explicit_ids = self._parse_visible_ids(
            os.environ.get("RLVR_VLLM_CUDA_VISIBLE_DEVICES") or os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
        )
        model_update_rank = getattr(self, "_model_update_rank", None)
        if isinstance(model_update_rank, str) and model_update_rank.isdigit():
            model_update_rank = int(model_update_rank)
        if not isinstance(model_update_rank, int):
            model_update_rank = None

        if explicit_ids and model_update_rank is not None:
            requested_gpu = explicit_ids[model_update_rank % len(explicit_ids)]

            current_visible = self._parse_visible_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
            if current_visible:
                try:
                    return current_visible.index(requested_gpu)
                except ValueError:
                    pass
            if 0 <= requested_gpu < device_count:
                return requested_gpu
            return model_update_rank % device_count

        if explicit_ids and torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                rank = int(torch.distributed.get_rank())
                requested_gpu = explicit_ids[rank % len(explicit_ids)]

                current_visible = self._parse_visible_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
                if current_visible:
                    try:
                        return current_visible.index(requested_gpu)
                    except ValueError:
                        pass
                if 0 <= requested_gpu < device_count:
                    return requested_gpu
                return rank % device_count
            except Exception:
                pass

        ray_assigned_index = _resolve_ray_assigned_cuda_index()
        if ray_assigned_index is not None and 0 <= ray_assigned_index < device_count:
            return ray_assigned_index

        # Prefer distributed rank when all devices are visible to avoid
        # multiple workers binding to cuda:0 in TP setups.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return int(torch.distributed.get_rank()) % device_count
            except Exception:
                pass

        device = getattr(self, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
            return int(device.index) % device_count
        if isinstance(device, int):
            return int(device) % device_count
        if isinstance(device, str) and device.startswith("cuda:"):
            suffix = device.split(":", maxsplit=1)[1]
            if suffix.isdigit():
                return int(suffix) % device_count
        return 0

    def _bind_cuda_device(self, torch) -> int:
        """Bind CUDA to the worker's assigned device before NCCL collectives."""
        device_index = self._resolve_cuda_index(torch)

        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
            device_index = int(torch.cuda.current_device())
        return device_index

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        use_ray=False,
        timeout_minutes=120,
    ):
        """Init torch process group for model weights update"""
        from datetime import timedelta

        import torch

        from open_instruct.vllm_utils import init_process_group

        print("init_process_group")
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_rank = int(rank)
        self._model_update_world_size = int(world_size)
        bound_cuda = self._bind_cuda_device(torch)
        print(
            "init_process_group cuda: "
            f"bound_cuda={bound_cuda}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"RLVR_VLLM_CUDA_VISIBLE_DEVICES={os.environ.get('RLVR_VLLM_CUDA_VISIBLE_DEVICES')}"
        )

        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            print("init_process_group else")
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                timeout=timedelta(minutes=timeout_minutes),
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        self._bind_cuda_device(torch)
        weight = torch.empty(shape, dtype=self.model_config.dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import os

        import torch

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        self._bind_cuda_device(torch)
        current_visible_index = int(torch.cuda.current_device())
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            visible_ids = [token.strip() for token in visible.split(",") if token.strip()]
            if current_visible_index < len(visible_ids) and visible_ids[current_visible_index].isdigit():
                physical_gpu_id = int(visible_ids[current_visible_index])
            else:
                physical_gpu_id = current_visible_index
        else:
            physical_gpu_id = current_visible_index

        handle = ipc_handles[physical_gpu_id]
        device_attr = getattr(self, "device", None)
        if isinstance(device_attr, torch.device) and device_attr.index is not None:
            device_id = int(device_attr.index)
        else:
            device_id = current_visible_index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
