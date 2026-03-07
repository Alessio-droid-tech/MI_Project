import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Tema per i grafici
sb.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Modifica dove prende i dati da tutti i modelli testati e non solo dai migliori dei due dataset
RESULTS = {
    'physionet_csp_lda': os.path.join("results", "PHY_CSP_LDA",        "phy_csp_lda_accuracies.npy"),
    'physionet_csp_svm': os.path.join("results", "PHYSIONET_CSP_SVM",  "physionet_CSP_SVM_accuracies.npy"),
    'physionet_riem_lr': os.path.join("results", "PHY_RI_LR",          "phy_ri_lr_accuracies.npy"),
    'physionet_fbr_svm': os.path.join("results", "PHY_FBR_SVM",        "phy_fbr_svm_accuracies.npy"),
    'bci_csp_lda':       os.path.join("results", "BCI_CSP_LDA",        "BCI_CSP_LDA_accuracies.npy"),
    'bci_csp_svm':       os.path.join("results", "BCI_CSP_SVM",        "bci_iv_csp_svm_accuracies.npy"),
    'bci_riem_lr':       os.path.join("results", "BCI_RI_LR",          "bci_RI_LR_accuracies.npy"),
    'bci_fbr_svm':       os.path.join("results", "BCI_FBR_SVM",        "bci_iv_accuracies.npy"),
}

# Creazione cartella per contenere immagini
PLOTS_DIR = os.path.join("results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load(key):
    return np.load(RESULTS[key]) * 100

def generate_plots():

    # GRAFICO 1 — Boxplot Naive vs Trained (Capitolo 4 / BCI Illiteracy)
    # Usa CSP + SVM come modello rappresentativo per entrambi
    physionet_best = load('physionet_csp_svm')
    bci_best       = load('bci_fbr_svm')

    plt.figure(figsize=(7, 6))
    sb.boxplot(data=[physionet_best, bci_best],
               palette=["skyblue", "lightgreen"], width=0.5)
    plt.xticks([0, 1],
               ['PhysioNet EEGMMIDB\n(Naive subjects, N=103)',
                'BCI Competition IV 2a\n(Trained subjects, N=9)'])
    plt.axhline(50.0, color='gray', linestyle=':', label='Chance level (50%)')
    plt.title("Subject-level Accuracy Distribution:\nNaive vs. Trained Subjects")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplot_naive_vs_trained.png"), dpi=150)
    plt.show()

 
    # GRAFICO 2 — Istogramma distribuzione PhysioNet CSP + SVM (Cap. 5.1)
    plt.figure(figsize=(8, 5))
    sb.histplot(physionet_best, bins=20, kde=True, color="skyblue")
    plt.axvline(np.mean(physionet_best), color='red', linestyle='--',
                label=f'Mean: {np.mean(physionet_best):.1f}%')
    plt.axvline(50.0, color='gray', linestyle=':',
                label='Chance level (50%)')
    plt.title("PhysioNet EEGMMIDB — Accuracy Distribution\nCSP + SVM, 103 subjects")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "physionet_accuracy_distribution.png"), dpi=150)
    plt.show()


    # GRAFICO 3 — Barplot per soggetto BCI IV, tutti i modelli (Cap. 5.2)
    subjects   = [f"A0{i}" for i in range(1, 10)]
    csp_lda    = load('bci_csp_lda')
    csp_svm    = load('bci_csp_svm')
    riem_lr    = load('bci_riem_lr')
    fbr_svm    = load('bci_fbr_svm')

    x     = np.arange(len(subjects))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, csp_lda, width, label='CSP+LDA',    color='#4C72B0')
    ax.bar(x - 0.5*width, csp_svm, width, label='CSP+SVM',    color='#DD8452')
    ax.bar(x + 0.5*width, riem_lr, width, label='Riem.+LR',   color='#55A868')
    ax.bar(x + 1.5*width, fbr_svm, width, label='FBR+SVM',    color='#C44E52')

    ax.axhline(50.0, color='gray', linestyle=':', label='Chance level (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("BCI Competition IV 2a — Per-subject Accuracy\nAll models, cross-session T→E")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bci_iv_per_subject_all_models.png"), dpi=150)
    plt.show()


    # GRAFICO 4 — Boxplot comparativo 4 modelli su BCI IV (Cap. 5.3)
    plt.figure(figsize=(8, 6))
    sb.boxplot(data=[csp_lda, csp_svm, riem_lr, fbr_svm],
               palette=['#4C72B0', '#DD8452', '#55A868', '#C44E52'],
               width=0.5)
    plt.xticks([0, 1, 2, 3],
               ['CSP+LDA', 'CSP+SVM', 'Riem.+LR', 'FBR+SVM'])
    plt.axhline(50.0, color='gray', linestyle=':', label='Chance level (50%)')
    plt.title("BCI Competition IV 2a — Model Comparison\nAccuracy distribution across 9 subjects")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bci_iv_model_comparison_boxplot.png"), dpi=150)
    plt.show()


    # GRAFICO 5 — Heatmap per soggetto × modello su BCI IV (Cap. 5.2/5.3)
    df = pd.DataFrame({
        'CSP+LDA':  csp_lda,
        'CSP+SVM':  csp_svm,
        'Riem.+LR': riem_lr,
        'FBR+SVM':  fbr_svm,
    }, index=subjects)

    plt.figure(figsize=(8, 6))
    sb.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
               vmin=50, vmax=100,
               linewidths=0.5, linecolor='gray')
    plt.title("BCI Competition IV 2a — Accuracy Heatmap\n(per subject × model, %)")
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bci_iv_accuracy_heatmap.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    generate_plots()