import matplotlib.pyplot as plt


def plot_results_grl(history, path_prefix):
    plt.figure()
    plt.plot(history["epoch"], history["train_total"], label="Train")
    plt.plot(history["epoch"], history["val_total"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total Loss (Train vs Val)")
    plt.legend()
    plt.savefig(path_prefix + "total.png", format="png")
    plt.close()

    task_titles = {
        "lan": "Lanes CE",
        "onw": "Oneway BCE",
        "wid": "Width Huber",
        "max": "Max Speed Huber",
        "min": "Min Speed Huber",
        "avg": "Avg Speed Huber",
    }
    for t, title in task_titles.items():
        plt.figure()
        plt.plot(history["epoch"], history["train_losses"][t], label="Train")
        plt.plot(history["epoch"], history["val_losses"][t], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.savefig(path_prefix + f"{t}.png", format="png")
        plt.close()

    metric_titles = {
        "lan_macro_f1": "Lanes Macro-F1",
        "onw_auroc": "Oneway AUROC",
        "wid_mae_m": "Width MAE (m)",
        "max_mae": "Max Speed MAE",
        "min_mae": "Min Speed MAE",
        "avg_mae": "Avg Speed MAE",
    }
    for m, title in metric_titles.items():
        plt.figure()
        plt.plot(history["metric_epoch"], history["train_metrics"][m], label="Train")
        plt.plot(history["metric_epoch"], history["val_metrics"][m], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(title)
        plt.legend()
        plt.savefig(path_prefix + f"{m}.png", format="png")
        plt.close()

