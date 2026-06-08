"""
Iterative Pruning Tools for Nomeroff Net
"""
import os
import copy
import torch
import torch_pruning as tp
import matplotlib.pyplot as plt

class IterativePruner:
    def __init__(self, model, example_inputs, eval_fn, train_fn, 
                 max_acc_drop=0.05, target_pruning_ratio=0.5, 
                 ignored_layers_builder=None,
                 save_dir='./pruning_results',
                 global_pruning=False,
                 isomorphic=False,
                 metric_direction="maximize",
                 unwrapped_parameters=None,
                 importance_criterion=None):
        """
        Args:
            model (nn.Module): The PyTorch or PyTorch Lightning model to prune.
            example_inputs (torch.Tensor): Dummy inputs for tracing the dependency graph.
            eval_fn (callable): A function `eval_fn(model)` that returns a scaler validation metric.
            train_fn (callable): A function `train_fn(model)` that finetunes the model for a few epochs.
                                 NOTE: Torch-Pruning explicitly requires initializing a NEW Optimizer 
                                 inside this function because the model's structural parameters change!
            max_acc_drop (float): Maximum allowed drop in the validation metric relative to the original model. 
            target_pruning_ratio (float): The ultimate target pruning ratio of channels by the last iteration.
            ignored_layers_builder (callable): A function `ignored_layers_builder(model)` returning a list of layers.
            importance_criterion: A `torch_pruning.importance` criterion. Defaults to GroupMagnitudeImportance(p=2).
            save_dir (str): Path to save checkpoints and metrics plot.
            isomorphic (bool): Enable isomorphic pruning.
            global_pruning (bool): Enable global pruning across all prunable groups.
            metric_direction (str): "maximize" (e.g., Accuracy, mAP) or "minimize" (e.g., CER, Loss).
        """
        self.model = model
        self.example_inputs = example_inputs
        self.eval_fn = eval_fn
        self.train_fn = train_fn
        self.max_acc_drop = max_acc_drop
        self.target_pruning_ratio = target_pruning_ratio
        self.ignored_layers_builder = ignored_layers_builder
        self.save_dir = save_dir
        self.isomorphic = isomorphic
        self.global_pruning = global_pruning
        
        if metric_direction not in ["maximize", "minimize"]:
            raise ValueError("metric_direction must be either 'maximize' or 'minimize'")
        self.metric_direction = metric_direction
        self.unwrapped_parameters = unwrapped_parameters
        
        if importance_criterion is None:
            self.importance = tp.importance.GroupMagnitudeImportance(p=2)
        else:
            self.importance = importance_criterion
            
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self, max_iters=10):
        """
        Run the iterative pruning process.
        Returns:
            most_compressed_valid_model: The most compressed pruned model within the metric drop limits.
            metrics: A dictionary of collected metrics over iterations.
        """
        print("Evaluating base model...")
        base_metric = self.eval_fn(self.model)
        
        # Determine MACs/Params explicitly without masking legitimate errors
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)

        metrics = {
            "iter": [0],
            "macs": [base_macs],
            "params": [base_nparams],
            "metric": [base_metric]
        }

        print(f"[Iter 0] Base Metric: {base_metric:.4f}, MACs: {base_macs / 1e9:.3f} G, Params: {base_nparams / 1e6:.3f} M")

        most_compressed_valid_model = copy.deepcopy(self.model)
        current_model = self.model

        ignored_layers = []
        if self.ignored_layers_builder is not None:
            ignored_layers = self.ignored_layers_builder(current_model)

        # 1. Initialize ONE Canonical Pruner outside the loop
        pruner = tp.pruner.BasePruner(
            current_model,
            self.example_inputs,
            importance=self.importance,
            pruning_ratio=self.target_pruning_ratio,
            ignored_layers=ignored_layers,
            iterative_steps=max_iters,
            isomorphic=self.isomorphic,
            global_pruning=self.global_pruning,
            unwrapped_parameters=self.unwrapped_parameters
        )

        for iter_idx in range(1, max_iters + 1):
            print(f"\n--- [Iter {iter_idx}/{max_iters}] Pruning ---")
            pruner.step()

            macs, nparams = tp.utils.count_ops_and_params(current_model, self.example_inputs)
            print(f"Pruned MACs: {macs / 1e9:.3f} G, Params: {nparams / 1e6:.3f} M")

            print(f"[Iter {iter_idx}] Evaluating after pruning...")
            metric_after_pruning = self.eval_fn(current_model)
            print(f"[Iter {iter_idx}] Metric after pruning: {metric_after_pruning:.4f} (Base: {base_metric:.4f})")
            
            metric = metric_after_pruning
            
            # Check drop relative to metric direction
            if self.metric_direction == "maximize":
                drop = base_metric - metric_after_pruning
            else:
                drop = metric_after_pruning - base_metric

            # Check if within error margin
            if drop <= self.max_acc_drop:
                print(f"Metric drop ({drop:.4f}) is within threshold ({self.max_acc_drop}). No fine-tuning needed.")
            else:
                if self.train_fn is not None:
                    print(f"[Iter {iter_idx}] Drop is too high! Fine-tuning pruned model...")
                    self.train_fn(current_model)
                    
                    print(f"[Iter {iter_idx}] Evaluating after fine-tuning...")
                    metric_after_finetune = self.eval_fn(current_model)
                    print(f"[Iter {iter_idx}] Metric after fine-tuning: {metric_after_finetune:.4f} (Base: {base_metric:.4f})")
                    metric = metric_after_finetune

            metrics["iter"].append(iter_idx)
            metrics["macs"].append(macs)
            metrics["params"].append(nparams)
            metrics["metric"].append(metric)

            # Re-check updated drop after fine-tuning
            if self.metric_direction == "maximize":
                final_drop = base_metric - metric
            else:
                final_drop = metric - base_metric

            # Check if this model is within acceptable drop threshold to be considered the "best" so far.
            if final_drop <= self.max_acc_drop:
                print(f"Metric drop ({final_drop:.4f}) is within threshold ({self.max_acc_drop}). This is the most compressed valid model so far!")
                most_compressed_valid_model = copy.deepcopy(current_model)
            else:
                print(f"Metric drop {final_drop:.4f} > max_drop {self.max_acc_drop}. Metric degraded, but continuing pruning...")

            checkpoint_path = os.path.join(self.save_dir, f"pruned_iter_{iter_idx}.pth")
            current_model.zero_grad()
            torch.save(current_model, checkpoint_path)

        print("\nPlotting metrics...")
        self._plot_metrics(metrics)

        print("Saving most compressed valid model configuration...")
        best_checkpoint_path = os.path.join(self.save_dir, "best_pruned_model.pth")
        most_compressed_valid_model.zero_grad()
        torch.save(most_compressed_valid_model, best_checkpoint_path)

        return most_compressed_valid_model, metrics

    def _plot_metrics(self, metrics):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if metrics["macs"][0] > 0:
            plt.plot(metrics["iter"], [m / 1e9 for m in metrics["macs"]], marker='o', label='MACs (G)')
        plt.plot(metrics["iter"], [p / 1e6 for p in metrics["params"]], marker='x', label='Params (M)')
        plt.xlabel("Pruning Iterations")
        plt.ylabel("Amount")
        plt.title("MACs & Params over iterations")
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(metrics["iter"], metrics["metric"], marker='o', label='Metric', color='green')
        plt.axhline(y=metrics["metric"][0], color='r', linestyle='--', label='Base Metric')
        if self.max_acc_drop > 0:
            if self.metric_direction == "maximize":
                plt.axhline(y=metrics["metric"][0] - self.max_acc_drop, color='orange', linestyle='--', label='Threshold')
            else:
                plt.axhline(y=metrics["metric"][0] + self.max_acc_drop, color='orange', linestyle='--', label='Threshold')
        plt.xlabel("Pruning Iterations")
        plt.ylabel("Metric")
        plt.title("Metric over iterations")
        plt.grid()
        plt.legend()

        plot_path = os.path.join(self.save_dir, "pruning_metrics.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Metrics plot saved to {plot_path}")
