import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Configurazione stile dei grafici
sb.set_theme(style="whitegrid", context="paper", font_scale=1.2)

RESULTS_DIR = "results"

def generate_plots():
    # Caricamento dei dati
    physionet_acc = np.load(os.path.join(RESULTS_DIR, "all_accuracies.npy")) * 100
    bci_iv_acc = np.load(os.path.join(RESULTS_DIR, "bci_iv_accuracies.npy")) * 100


    # GRAFICO 1: Istogramma PhysioNet
    plt.figure(figsize=(8, 5))
    sb.histplot(physionet_acc, bins=20, kde=True, color="skyblue")
    plt.axvline(np.mean(physionet_acc), color='red', linestyle='--', label=f'Mean: {np.mean(physionet_acc):.1f}%')
    plt.axvline(50.0, color='gray', linestyle=':', label='Chance Level (50%)')
    plt.title("PhysioNet Dataset: Accuracy Distribution (Naive Subjects)")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Subjects")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # GRAFICO 2: Barplot BCI IV 2a
    plt.figure(figsize=(8, 5))
    subjects = [f"A0{i}T" for i in range(1, 10)]
    sb.barplot(x=subjects, y=bci_iv_acc, palette="viridis")
    plt.axhline(np.mean(bci_iv_acc), color='red', linestyle='--', label=f'Mean: {np.mean(bci_iv_acc):.1f}%')
    plt.axhline(50.0, color='gray', linestyle=':', label='Chance Level')
    plt.ylim(0, 100)
    plt.title("BCI Competition IV 2a: Individual Subject Accuracy (Trained)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # GRAFICO 3: Boxplot Comparativo
    plt.figure(figsize=(7, 6))
    data = [physionet_acc, bci_iv_acc]
    sb.boxplot(data=data, palette=["skyblue", "lightgreen"], width=0.5)
    plt.xticks([0, 1], ['PhysioNet\n(Naive, N=103)', 'BCIC IV 2a\n(Trained, N=9)'])
    plt.axhline(50.0, color='gray', linestyle=':', label='Chance Level')
    plt.title("Performance Comparison: Naive vs Trained Subjects")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_plots()