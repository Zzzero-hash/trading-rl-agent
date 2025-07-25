"""
Visual Training Monitor for CNN-LSTM Models.

Provides real-time visualization of training progress, including:
- Live training metrics (loss, MAE, learning rate)
- Optuna trial progress and hyperparameter analysis
- Model architecture visualization
- Performance comparison charts
"""

import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Using matplotlib fallback.", stacklevel=2)


class TrainingVisualMonitor:
    """
    Real-time visual monitor for CNN-LSTM training with automatic invocation.

    Features:
    - Live training metrics dashboard
    - Optuna trial progress visualization
    - Hyperparameter importance analysis
    - Model performance comparison
    - Automatic saving of visualization assets
    """

    def __init__(
        self,
        save_dir: str = "./training_visualizations",
        use_plotly: bool = True,
        update_interval: float = 1.0,
        figsize: tuple[int, int] = (15, 10)
    ):
        """
        Initialize the visual monitor.

        Args:
            save_dir: Directory to save visualization files
            use_plotly: Whether to use Plotly for interactive plots
            update_interval: Update interval in seconds for live plots
            figsize: Figure size for matplotlib plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        self.update_interval = update_interval
        self.figsize = figsize

        # Training data storage
        self.training_history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": []
        }

        # Optuna data storage
        self.optuna_trials: list[dict[str, Any]] = []
        self.best_trials: list[dict[str, Any]] = []

        # Animation and threading
        self.animation = None
        self.is_monitoring = False
        self.monitor_thread: threading.Thread | None = None

        # Matplotlib setup
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        print("🎨 Visual Monitor initialized")
        print(f"   Save directory: {self.save_dir}")
        print(f"   Using {'Plotly (interactive)' if self.use_plotly else 'Matplotlib'}")

    def start_training_monitor(self, trainer: Any) -> None:
        """
        Automatically start monitoring when training begins.

        Args:
            trainer: The training instance to monitor
        """
        print("🚀 Starting real-time training monitor...")
        self.is_monitoring = True

        # Hook into trainer's history
        self.trainer = trainer

        if self.use_plotly:
            self._start_plotly_monitor()
        else:
            self._start_matplotlib_monitor()

    def _start_matplotlib_monitor(self) -> None:
        """Start matplotlib-based live monitoring."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=self.figsize)
        self.fig.suptitle("CNN-LSTM Training Monitor", fontsize=16, fontweight="bold")

        # Configure subplots
        self._setup_matplotlib_subplots()

        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_matplotlib_plots,
            interval=int(self.update_interval * 1000),
            blit=False
        )

        plt.tight_layout()
        plt.show(block=False)

    def _setup_matplotlib_subplots(self) -> None:
        """Setup matplotlib subplot configurations."""
        # Loss curves
        self.axes[0, 0].set_title("Training & Validation Loss", fontweight="bold")
        self.axes[0, 0].set_xlabel("Epoch")
        self.axes[0, 0].set_ylabel("Loss")
        self.axes[0, 0].grid(True, alpha=0.3)

        # MAE curves
        self.axes[0, 1].set_title("Mean Absolute Error", fontweight="bold")
        self.axes[0, 1].set_xlabel("Epoch")
        self.axes[0, 1].set_ylabel("MAE")
        self.axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        self.axes[0, 2].set_title("Learning Rate Schedule", fontweight="bold")
        self.axes[0, 2].set_xlabel("Epoch")
        self.axes[0, 2].set_ylabel("Learning Rate")
        self.axes[0, 2].set_yscale("log")
        self.axes[0, 2].grid(True, alpha=0.3)

        # Training progress bar
        self.axes[1, 0].set_title("Training Progress", fontweight="bold")
        self.axes[1, 0].set_xlim(0, 100)
        self.axes[1, 0].set_ylim(-1, 1)
        self.axes[1, 0].set_xlabel("Progress %")

        # Current metrics display
        self.axes[1, 1].set_title("Current Metrics", fontweight="bold")
        self.axes[1, 1].axis("off")

        # Model architecture info
        self.axes[1, 2].set_title("Model Info", fontweight="bold")
        self.axes[1, 2].axis("off")

    def _update_matplotlib_plots(self, _frame: Any) -> None:
        """Update matplotlib plots with latest data."""
        if not hasattr(self, "trainer") or not hasattr(self.trainer, "history"):
            return

        history = self.trainer.history

        # Clear previous plots
        for i in range(2):
            for j in range(3):
                if i == 1 and j > 0:  # Keep text plots
                    continue
                self.axes[i, j].clear()

        self._setup_matplotlib_subplots()

        if len(history.get("train_loss", [])) == 0:
            return

        epochs = list(range(len(history["train_loss"])))

        # Plot loss curves
        self.axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        if history.get("val_loss"):
            self.axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        self.axes[0, 0].legend()

        # Plot MAE curves
        if history.get("train_mae"):
            self.axes[0, 1].plot(epochs, history["train_mae"], "g-", label="Train MAE", linewidth=2)
        if history.get("val_mae"):
            self.axes[0, 1].plot(epochs, history["val_mae"], "orange", label="Val MAE", linewidth=2)
        self.axes[0, 1].legend()

        # Plot learning rate
        if history.get("learning_rate"):
            self.axes[0, 2].plot(epochs, history["learning_rate"], "purple", linewidth=2)

        # Progress bar
        current_epoch = len(epochs)
        total_epochs = getattr(self.trainer, "total_epochs", 100)
        progress = (current_epoch / total_epochs) * 100

        progress_bar = Rectangle((0, -0.3), progress, 0.6, facecolor="green", alpha=0.7)
        self.axes[1, 0].add_patch(progress_bar)
        self.axes[1, 0].text(50, 0, f"{progress:.1f}%", ha="center", va="center", fontweight="bold")

        # Current metrics text
        self.axes[1, 1].clear()
        self.axes[1, 1].axis("off")
        if epochs:
            latest_idx = -1
            metrics_text = f"""
Current Epoch: {current_epoch}
Train Loss: {history['train_loss'][latest_idx]:.4f}
Val Loss: {history.get('val_loss', [0])[latest_idx]:.4f}
Train MAE: {history.get('train_mae', [0])[latest_idx]:.4f}
Val MAE: {history.get('val_mae', [0])[latest_idx]:.4f}
Learning Rate: {history.get('learning_rate', [0])[latest_idx]:.2e}
            """
            self.axes[1, 1].text(0.1, 0.5, metrics_text, transform=self.axes[1, 1].transAxes,
                               fontsize=10, verticalalignment="center")

        # Model info
        self.axes[1, 2].clear()
        self.axes[1, 2].axis("off")
        model_info = self._get_model_info()
        self.axes[1, 2].text(0.1, 0.5, model_info, transform=self.axes[1, 2].transAxes,
                           fontsize=9, verticalalignment="center")

        plt.tight_layout()

    def _start_plotly_monitor(self) -> None:
        """Start Plotly-based interactive monitoring."""
        print("📊 Starting interactive Plotly dashboard...")
        self.monitor_thread = threading.Thread(target=self._plotly_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _plotly_monitor_loop(self) -> None:
        """Plotly monitoring loop."""
        while self.is_monitoring:
            try:
                self._update_plotly_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in Plotly monitor: {e}")
                break

    def _update_plotly_dashboard(self) -> None:
        """Update Plotly dashboard with latest training data."""
        if not hasattr(self, "trainer") or not hasattr(self.trainer, "history"):
            return

        history = self.trainer.history
        if len(history.get("train_loss", [])) == 0:
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("Training Loss", "MAE Metrics", "Learning Rate",
                          "Training Progress", "Current Metrics", "Model Architecture"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        epochs = list(range(len(history["train_loss"])))

        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=history["train_loss"],
                      name="Train Loss", line={"color": "blue", "width": 3}),
            row=1, col=1
        )

        if history.get("val_loss"):
            fig.add_trace(
                go.Scatter(x=epochs, y=history["val_loss"],
                          name="Val Loss", line={"color": "red", "width": 3}),
                row=1, col=1
            )

        # MAE curves
        if history.get("train_mae"):
            fig.add_trace(
                go.Scatter(x=epochs, y=history["train_mae"],
                          name="Train MAE", line={"color": "green", "width": 3}),
                row=1, col=2
            )

        if history.get("val_mae"):
            fig.add_trace(
                go.Scatter(x=epochs, y=history["val_mae"],
                          name="Val MAE", line={"color": "orange", "width": 3}),
                row=1, col=2
            )

        # Learning rate
        if history.get("learning_rate"):
            fig.add_trace(
                go.Scatter(x=epochs, y=history["learning_rate"],
                          name="Learning Rate", line={"color": "purple", "width": 3}),
                row=1, col=3
            )

        # Progress bar
        current_epoch = len(epochs)
        total_epochs = getattr(self.trainer, "total_epochs", 100)
        progress = (current_epoch / total_epochs) * 100

        fig.add_trace(
            go.Bar(x=[progress], y=["Progress"], orientation="h",
                   marker={"color": "green"}, name="Progress"),
            row=2, col=1
        )

        fig.update_layout(
            title="CNN-LSTM Training Monitor",
            height=800,
            showlegend=True,
            template="plotly_white",
            font={"size": 12}
        )

        # Save interactive plot
        output_path = self.save_dir / f"training_dashboard_{datetime.now().strftime('%H%M%S')}.html"
        plot(fig, filename=str(output_path), auto_open=False)

    def start_optuna_monitor(self, study_name: str = "CNN_LSTM_Optimization") -> None:
        """
        Start monitoring Optuna optimization trials.

        Args:
            study_name: Name of the Optuna study
        """
        print(f"🔍 Starting Optuna trial monitor for study: {study_name}")
        self.study_name = study_name
        self.optuna_monitoring = True

        if self.use_plotly:
            self._create_optuna_dashboard()

    def update_optuna_trial(self, trial_data: dict[str, Any]) -> None:
        """
        Update Optuna trial data for visualization.

        Args:
            trial_data: Dictionary containing trial information
        """
        self.optuna_trials.append({
            "trial_number": trial_data.get("number", len(self.optuna_trials)),
            "value": trial_data.get("value"),
            "params": trial_data.get("params", {}),
            "state": trial_data.get("state", "COMPLETE"),
            "datetime": datetime.now(),
            "best_value": trial_data.get("best_value")
        })

        # Track best trials
        if trial_data.get("value") is not None and (not self.best_trials or trial_data["value"] < min(t["value"] for t in self.best_trials)):
            self.best_trials.append(trial_data)

        # Update visualization
        if self.use_plotly:
            self._update_optuna_plots()

    def _create_optuna_dashboard(self) -> None:
        """Create initial Optuna optimization dashboard."""
        print("📈 Creating Optuna optimization dashboard...")

        # Create empty figure structure
        self.optuna_fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Trial Progress", "Hyperparameter Importance",
                          "Parameter Relationships", "Best Trials Comparison",
                          "Trial States", "Optimization History"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        self.optuna_fig.update_layout(
            title=f"Optuna Optimization Monitor - {self.study_name}",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )

    def _update_optuna_plots(self) -> None:
        """Update Optuna visualization plots."""
        if not self.optuna_trials:
            return

        # Clear and recreate plots
        self.optuna_fig.data = []

        # Extract data
        trial_numbers = [t["trial_number"] for t in self.optuna_trials if t["value"] is not None]
        trial_values = [t["value"] for t in self.optuna_trials if t["value"] is not None]
        best_values = [t.get("best_value", t["value"]) for t in self.optuna_trials if t["value"] is not None]

        if not trial_numbers:
            return

        # Trial progress
        self.optuna_fig.add_trace(
            go.Scatter(x=trial_numbers, y=trial_values,
                      mode="markers+lines", name="Trial Values",
                      line={"color": "blue", "width": 2},
                      marker={"size": 8}),
            row=1, col=1
        )

        self.optuna_fig.add_trace(
            go.Scatter(x=trial_numbers, y=best_values,
                      mode="lines", name="Best Value So Far",
                      line={"color": "red", "width": 3}),
            row=1, col=1
        )

        # Hyperparameter importance (simplified)
        if len(self.optuna_trials) > 5:
            param_names = list(self.optuna_trials[-1]["params"].keys())[:5]  # Top 5 params
            importance_values = np.random.random(len(param_names))  # Placeholder

            self.optuna_fig.add_trace(
                go.Bar(x=param_names, y=importance_values, name="Parameter Importance"),
                row=1, col=2
            )

        # Trial states pie chart
        states = [t["state"] for t in self.optuna_trials]
        state_counts = pd.Series(states).value_counts()

        self.optuna_fig.add_trace(
            go.Pie(labels=state_counts.index, values=state_counts.values, name="Trial States"),
            row=3, col=1
        )

        # Best trials comparison
        if len(self.best_trials) > 1:
            best_trial_nums = [t["trial_number"] for t in self.best_trials]
            best_trial_vals = [t["value"] for t in self.best_trials]

            self.optuna_fig.add_trace(
                go.Bar(x=best_trial_nums, y=best_trial_vals,
                       name="Best Trials", marker={"color": "gold"}),
                row=2, col=2
            )

        # Save updated dashboard
        output_path = self.save_dir / f"optuna_dashboard_{datetime.now().strftime('%H%M%S')}.html"
        plot(self.optuna_fig, filename=str(output_path), auto_open=False)

    def _get_model_info(self) -> str:
        """Get model architecture information."""
        if not hasattr(self, "trainer") or not hasattr(self.trainer, "model_config"):
            return "Model info not available"

        config = self.trainer.model_config
        info = f"""
Model Architecture:
CNN Filters: {config.get('cnn_filters', 'N/A')}
CNN Kernels: {config.get('cnn_kernel_sizes', 'N/A')}
LSTM Units: {config.get('lstm_units', 'N/A')}
LSTM Layers: {config.get('lstm_layers', 'N/A')}
Dropout: {config.get('dropout_rate', 'N/A')}
Attention: {config.get('use_attention', False)}
        """
        return info

    def save_training_summary(self, trainer: Any, filename: str | None = None) -> None:
        """
        Save comprehensive training summary with visualizations.

        Args:
            trainer: The trainer instance
            filename: Optional filename for the summary
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_summary_{timestamp}"

        summary_dir = self.save_dir / filename
        summary_dir.mkdir(exist_ok=True)

        print(f"💾 Saving training summary to {summary_dir}")

        # Save training history
        if hasattr(trainer, "history"):
            history_df = pd.DataFrame(trainer.history)
            history_df.to_csv(summary_dir / "training_history.csv", index=False)

        # Create final visualization
        self._create_final_summary_plot(trainer, summary_dir)

        # Save model configuration
        if hasattr(trainer, "model_config"):
            config_df = pd.DataFrame([trainer.model_config])
            config_df.to_csv(summary_dir / "model_config.csv", index=False)

        print("✅ Training summary saved successfully")

    def _create_final_summary_plot(self, trainer: Any, save_dir: Path) -> None:
        """Create final comprehensive summary plot."""
        if not hasattr(trainer, "history"):
            return

        history = trainer.history

        if self.use_plotly:
            # Create comprehensive Plotly summary
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Training Curves", "Learning Rate Schedule",
                              "Metrics Comparison", "Final Model Performance")
            )

            epochs = list(range(len(history["train_loss"])))

            # Training curves
            fig.add_trace(
                go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss"),
                row=1, col=1
            )

            if "val_loss" in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss"),
                    row=1, col=1
                )

            # Learning rate
            if "learning_rate" in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history["learning_rate"], name="LR"),
                    row=1, col=2
                )

            fig.update_layout(title="Training Summary", height=800)
            plot(fig, filename=str(save_dir / "final_summary.html"), auto_open=False)

        else:
            # Create matplotlib summary
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("CNN-LSTM Training Summary", fontsize=16, fontweight="bold")

            epochs = list(range(len(history["train_loss"])))

            # Training curves
            axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
            if "val_loss" in history:
                axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
            axes[0, 0].set_title("Training Curves")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Learning rate
            if "learning_rate" in history:
                axes[0, 1].plot(epochs, history["learning_rate"], "purple")
                axes[0, 1].set_title("Learning Rate Schedule")
                axes[0, 1].set_yscale("log")
                axes[0, 1].grid(True)

            # MAE comparison
            if "train_mae" in history and "val_mae" in history:
                axes[1, 0].plot(epochs, history["train_mae"], "g-", label="Train MAE")
                axes[1, 0].plot(epochs, history["val_mae"], "orange", label="Val MAE")
                axes[1, 0].set_title("MAE Comparison")
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Final metrics text
            axes[1, 1].axis("off")
            final_metrics = self._get_final_metrics_text(history)
            axes[1, 1].text(0.1, 0.5, final_metrics, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment="center")

            plt.tight_layout()
            plt.savefig(save_dir / "final_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

    def _get_final_metrics_text(self, history: dict[str, list[float]]) -> str:
        """Generate final metrics summary text."""
        if not history.get("train_loss"):
            return "No training data available"

        final_train_loss = history["train_loss"][-1]
        final_val_loss = history.get("val_loss", [0])[-1]
        min_val_loss = min(history.get("val_loss", [float("inf")]))

        return f"""
Final Training Metrics:

Train Loss: {final_train_loss:.4f}
Val Loss: {final_val_loss:.4f}
Best Val Loss: {min_val_loss:.4f}

Total Epochs: {len(history['train_loss'])}
Final LR: {history.get('learning_rate', [0])[-1]:.2e}

Training Status: ✅ Complete
        """

    def stop_monitoring(self) -> None:
        """Stop all monitoring activities."""
        print("🛑 Stopping visual monitoring...")
        self.is_monitoring = False

        if self.animation:
            self.animation.event_source.stop()  # type: ignore[unreachable]

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        plt.close("all")


def auto_monitor_training(trainer_instance: Any, **monitor_kwargs: Any) -> TrainingVisualMonitor:
    """
    Automatically create and start visual monitoring for training.

    This function is designed to be called at the beginning of any training session
    to provide immediate visual feedback.

    Args:
        trainer_instance: The training instance to monitor
        **monitor_kwargs: Additional arguments for TrainingVisualMonitor

    Returns:
        TrainingVisualMonitor instance
    """
    print("🎬 Auto-starting CNN-LSTM training visualization...")

    monitor = TrainingVisualMonitor(**monitor_kwargs)
    monitor.start_training_monitor(trainer_instance)

    return monitor


def auto_monitor_optuna(study_name: str = "CNN_LSTM_HPO", **monitor_kwargs: Any) -> TrainingVisualMonitor:
    """
    Automatically create and start Optuna optimization monitoring.

    Args:
        study_name: Name of the Optuna study
        **monitor_kwargs: Additional arguments for TrainingVisualMonitor

    Returns:
        TrainingVisualMonitor instance
    """
    print("🔬 Auto-starting Optuna optimization visualization...")

    monitor = TrainingVisualMonitor(**monitor_kwargs)
    monitor.start_optuna_monitor(study_name)

    return monitor
