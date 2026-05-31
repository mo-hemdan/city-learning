import numpy as np
import matplotlib.pyplot as plt

def plot_avgspeed_nans_per_bin(data, filename):
    nan_counts = np.sum(np.isnan(data.y_avg_speed.cpu().numpy()), axis=1)

    plt.figure(figsize=(10, 4))
    plt.hist(nan_counts, bins=50)
    plt.xlabel("Number of NaN timesteps")
    plt.ylabel("Number of edges")
    plt.title("Distribution of NaN counts across edges")
    plt.savefig(filename, format='png')
    
    
    
# =========================
# 11) Plots
# =========================

def plot_results(history, path_prefix):
    plt.figure()
    plt.plot(history["epoch"], history["train_total"], label="Train")
    plt.plot(history["epoch"], history["val_total"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total Loss (Train vs Val)")
    plt.legend()
    plt.savefig(path_prefix+f"total.png", format='png')

    task_titles = {
        "hwy": "Highway CE",
        "lan": "Lanes CE",
        "onw": "Oneway BCE",
        "wid": "Width Huber",
        "max": "Max Speed Huber",
        "min": "Min Speed Huber",
    }
    for t in ["hwy", "lan", "onw", "wid", "max", "min"]:
        plt.figure()
        plt.plot(history["epoch"], history["train_losses"][t], label="Train")
        plt.plot(history["epoch"], history["val_losses"][t], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{task_titles[t]} (masked-only)")
        plt.legend()
        plt.savefig(path_prefix+f"{t}.png", format='png')

    metric_titles = {
        "hwy_macro_f1": "Highway Macro-F1 (masked)",
        "lan_macro_f1": "Lanes Macro-F1 (masked)",
        "onw_auroc": "Oneway AUROC (masked)",
        "wid_mae_m": "Width MAE (m, masked)",
        "max_mae": "Max Speed MAE (masked)",
        "min_mae": "Min Speed MAE (masked)",
    }
    for m in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae"]:
        plt.figure()
        plt.plot(history["metric_epoch"], history["train_metrics"][m], label="Train")
        plt.plot(history["metric_epoch"], history["val_metrics"][m], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(metric_titles[m])
        plt.legend()
        plt.savefig(path_prefix+f"{m}.png", format='png')
