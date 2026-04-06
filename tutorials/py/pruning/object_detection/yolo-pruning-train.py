"""
LINKS
 * https://www.ultralytics.com/blog/pruning-and-quantization-in-computer-vision-a-quick-guide
 * https://github.com heyongxin233/YOLO-Pruning-RKNN
 * https://y-t-g.github.io/tutorials/yolo-prune/
"""
import argparse
import io
import math
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 pruning + fine-tuning with NVIDIA ModelOpt")
    parser.add_argument("--dataset-yaml", required=True, help="Path to dataset yaml")
    parser.add_argument("--weights", required=True, help="Path to source YOLO weights")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index inside container")
    parser.add_argument("--target-params", default="80%", help='Target params constraint, e.g. "80%"')
    parser.add_argument("--epochs", type=int, default=1, help="Fine-tuning epochs after pruning")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--optimizer", default="SGD", help='Optimizer for fine-tuning, e.g. "SGD" or "AdamW"')
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate for pruned model fine-tuning")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD/optimizer")
    parser.add_argument("--warmup_epochs", type=float, default=0.0, help="Warmup epochs")
    parser.add_argument("--amp", action="store_true", help="Use AMP (disabled by default for stability)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--project", default="runs/prune", help="Project dir")
    parser.add_argument("--name", default="yolo11_pruned", help="Run name")
    parser.add_argument("--max-iter-data-loader", type=int, default=20, help="FastNAS loader iterations")
    return parser.parse_args()


@contextmanager
def patch_ultralytics_for_modelopt():
    """
    Patch Ultralytics to support:
    1) saving custom ModelOpt checkpoints
    2) saving a parallel stock-like rebuilt checkpoint
    3) reloading the just-trained model without fragile post-hoc reconstruction
    """
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn import tasks as yolo_tasks
    from ultralytics.engine import model as yolo_engine_model
    from ultralytics.utils import LOGGER
    import modelopt.torch.opt as mto

    original_save_model = BaseTrainer.save_model
    original_tasks_attempt_load_one_weight = getattr(yolo_tasks, "attempt_load_one_weight", None)
    original_engine_attempt_load_one_weight = getattr(yolo_engine_model, "attempt_load_one_weight", None)

    # Keep the live pruned model in memory so model.train() can finish cleanly
    live_model_cache = {"model": None}

    def _save_standard_ultralytics_ckpt(path: Path, model_obj, trainer):
        """
        Save a stock-like Ultralytics checkpoint using the already-correct live pruned model.
        """
        model_to_save = deepcopy(model_obj).half()

        # Remove non-picklable temporary lambda injected for ModelOpt
        if hasattr(model_to_save, "is_fused"):
            try:
                attr = getattr(model_to_save, "is_fused")
                if callable(attr) and getattr(attr, "__name__", "") == "<lambda>":
                    delattr(model_to_save, "is_fused")
            except Exception:
                pass

        rebuilt_ckpt = {
            "epoch": trainer.epoch,
            "best_fitness": trainer.best_fitness,
            "model": model_to_save,
            "ema": None,
            "updates": trainer.ema.updates if getattr(trainer, "ema", None) is not None else 0,
            "optimizer": None,
            "train_args": vars(trainer.args),
            "train_metrics": {**trainer.metrics, **{"fitness": trainer.fitness}},
            "date": datetime.now().isoformat(),
            "version": "custom-rebuilt",
        }

        # Use atomic replacement to avoid 246-byte corrupted files
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(rebuilt_ckpt, str(tmp_path))
        tmp_path.replace(path)

    def patched_save_model(self):
        """
        Save:
        - custom ModelOpt checkpoint (best.pt / last.pt)
        - standard rebuilt checkpoint (best_rebuilt.pt / last_rebuilt.pt)
        """
        from ultralytics.utils import torch_utils as tu
        from ultralytics import __version__

        unwrap_model = getattr(tu, "unwrap_model", None)
        if unwrap_model is None:
            unwrap_model = getattr(tu, "de_parallel", lambda m: m)

        convert_optimizer_state_dict_to_fp16 = getattr(
            tu,
            "convert_optimizer_state_dict_to_fp16",
            lambda x: x,
        )

        buffer = io.BytesIO()

        base_model = self.ema.ema if getattr(self, "ema", None) is not None else self.model
        model = deepcopy(unwrap_model(base_model))

        # Cache the live pruned model for later reload inside model.train()
        live_model_cache["model"] = deepcopy(model).float().cpu()

        extras = {}
        if hasattr(model, "_modelopt_state"):
            extras = {
                "modelopt_state": mto.modelopt_state(model),
                "state_dict": model.state_dict(),
                "model_class": model.__class__,
                "yaml": model.yaml,
                "names": model.names,
                "nc": model.nc,
            }

        custom_ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": None if extras else model.half(),
            "updates": self.ema.updates if getattr(self, "ema", None) is not None else 0,
            "optimizer": convert_optimizer_state_dict_to_fp16(self.optimizer.state_dict()),
            "scaler": self.scaler.state_dict() if getattr(self, "scaler", None) is not None else None,
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "date": datetime.now().isoformat(),
            "version": __version__,
            **extras,
        }

        torch.save(custom_ckpt, buffer)
        serialized = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)

        # Save custom checkpoints
        self.last.write_bytes(serialized)
        custom_last = self.last

        # Save rebuilt last checkpoint
        rebuilt_last = self.wdir / "last_rebuilt.pt"
        try:
            _save_standard_ultralytics_ckpt(rebuilt_last, model, self)
        except Exception as e:
            LOGGER.warning(f"Failed to save rebuilt last checkpoint: {e}")

        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized)
            custom_best = self.best

            rebuilt_best = self.wdir / "best_rebuilt.pt"
            try:
                _save_standard_ultralytics_ckpt(rebuilt_best, model, self)
            except Exception as e:
                LOGGER.warning(f"Failed to save rebuilt best checkpoint: {e}")

    def patched_attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
        """
        Intercept the post-train reload inside model.train().

        Strategy:
        1) If this is our custom ModelOpt checkpoint, do NOT try to rebuild from modelopt_state.
           Instead:
             - prefer live_model_cache if available
             - otherwise fall back to sibling *_rebuilt.pt
        2) If this is a normal checkpoint, delegate to original loader.
        """
        weight = str(weight)
        ckpt = torch.load(weight, map_location="cpu")

        is_custom_modelopt_ckpt = ckpt.get("model") is None and "modelopt_state" in ckpt
        if is_custom_modelopt_ckpt:
            # First choice: use live in-memory model
            if live_model_cache["model"] is not None:
                model = deepcopy(live_model_cache["model"])
                model = model.to(device).float() if device is not None else model.float()
                model.args = ckpt.get("train_args", {})
                model.pt_path = weight

                guess_model_task = getattr(yolo_tasks, "guess_model_task", None)
                if callable(guess_model_task):
                    try:
                        model.task = guess_model_task(model)
                    except Exception:
                        pass

                return model, ckpt

            # Second choice: try sibling rebuilt checkpoint from disk
            weight_path = Path(weight)
            rebuilt_path = weight_path.with_name(f"{weight_path.stem}_rebuilt.pt")
            if rebuilt_path.exists() and original_tasks_attempt_load_one_weight is not None:
                return original_tasks_attempt_load_one_weight(
                    str(rebuilt_path), device=device, inplace=inplace, fuse=fuse
                )

            raise ValueError(
                "Custom ModelOpt checkpoint detected, but no live model cache and no rebuilt checkpoint found."
            )

        # Normal stock checkpoint -> delegate to original loader if available
        if original_tasks_attempt_load_one_weight is not None:
            return original_tasks_attempt_load_one_weight(
                weight, device=device, inplace=inplace, fuse=fuse
            )

        # Very defensive fallback
        model = ckpt.get("ema") or ckpt.get("model")
        if model is None:
            raise ValueError("Could not load model from checkpoint.")
        model = model.to(device).float() if device is not None else model.float()
        return model, ckpt

    BaseTrainer.save_model = patched_save_model

    if original_tasks_attempt_load_one_weight is not None:
        yolo_tasks.attempt_load_one_weight = patched_attempt_load_one_weight

    if original_engine_attempt_load_one_weight is not None:
        yolo_engine_model.attempt_load_one_weight = patched_attempt_load_one_weight

    try:
        yield
    finally:
        BaseTrainer.save_model = original_save_model

        if original_tasks_attempt_load_one_weight is not None:
            yolo_tasks.attempt_load_one_weight = original_tasks_attempt_load_one_weight

        if original_engine_attempt_load_one_weight is not None:
            yolo_engine_model.attempt_load_one_weight = original_engine_attempt_load_one_weight


def make_pruned_trainer(base_trainer_cls, target_params: str, max_iter_data_loader: int):
    class PrunedTrainer(base_trainer_cls):
        def _setup_train(self, world_size):
            """
            Override train setup to:
            1) run standard Ultralytics setup
            2) apply ModelOpt FastNAS pruning
            3) rebuild optimizer / scheduler / EMA for the pruned model
            """
            import torchprofile.profile as tp_profile

            # Compatibility shim for old ModelOpt FLOPs/params internals
            if not hasattr(tp_profile, "handlers") and hasattr(tp_profile, "HANDLER_MAP"):
                tp_profile.handlers = list(tp_profile.HANDLER_MAP.items())

            import modelopt.torch.prune as mtp
            from ultralytics.utils.torch_utils import ModelEMA
            from ultralytics.utils import LOGGER

            super()._setup_train(world_size)

            def collect_func(batch):
                return self.preprocess_batch(batch)["img"]

            def score_func(candidate_model):
                candidate_model.eval()

                orig_save = self.validator.args.save
                orig_plots = self.validator.args.plots
                orig_verbose = self.validator.args.verbose

                self.validator.args.save = False
                self.validator.args.plots = False
                self.validator.args.verbose = False
                self.validator.is_coco = False

                metrics = self.validator(model=candidate_model)

                self.validator.args.save = orig_save
                self.validator.args.plots = orig_plots
                self.validator.args.verbose = orig_verbose

                return metrics["fitness"]

            LOGGER.info(f"Applying ModelOpt pruning with target params: {target_params}")

            # Some ModelOpt paths expect model.is_fused()
            self.model.is_fused = lambda: True

            self.model, prune_info = mtp.prune(
                model=self.model,
                mode="fastnas",
                constraints={"params": target_params},
                dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
                config={
                    "score_func": score_func,
                    "data_loader": self.train_loader,
                    "collect_func": collect_func,
                    "checkpoint": "modelopt_fastnas_search_checkpoint.pth",
                    "max_iter_data_loader": max_iter_data_loader,
                },
            )

            LOGGER.info(f"Prune info: {prune_info}")

            # Make post-pruning fine-tuning more stable: force SGD and low LR
            self.args.optimizer = getattr(self.args, "optimizer", "SGD") or "SGD"
            self.args.lr0 = min(float(getattr(self.args, "lr0", 1e-4)), 1e-4)
            self.args.momentum = float(getattr(self.args, "momentum", 0.9))
            self.args.warmup_epochs = float(getattr(self.args, "warmup_epochs", 0.0))

            # Remove temporary non-picklable helper used only for ModelOpt paths
            if hasattr(self.model, "is_fused"):
                try:
                    attr = getattr(self.model, "is_fused")
                    if callable(attr) and getattr(attr, "__name__", "") == "<lambda>":
                        delattr(self.model, "is_fused")
                except Exception:
                    pass

            self.model.to(self.device)
            self.ema = ModelEMA(self.model)

            weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs

            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
            self._setup_scheduler()

            LOGGER.info("Pruned model setup complete. Starting fine-tuning.")

        def final_eval(self):
            """
            Skip stock final_eval(), because stock strip_optimizer()
            expects ckpt['model'] to be a real model object in best.pt.
            """
            from ultralytics.utils import LOGGER

            LOGGER.info("Skipping stock final_eval() for ModelOpt custom checkpoint.")
            return

    return PrunedTrainer


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    with patch_ultralytics_for_modelopt():
        model = YOLO(str(weights_path))

        base_trainer_cls = model.task_map[model.task]["trainer"]
        trainer_cls = make_pruned_trainer(
            base_trainer_cls=base_trainer_cls,
            target_params=args.target_params,
            max_iter_data_loader=args.max_iter_data_loader,
        )

        model.train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            lr0=args.lr0,
            momentum=args.momentum,
            warmup_epochs=args.warmup_epochs,
            amp=args.amp,
            weight_decay=args.weight_decay,
            device=args.gpu,
            trainer=trainer_cls,
            project=args.project,
            name=args.name,
        )


if __name__ == "__main__":
    main()