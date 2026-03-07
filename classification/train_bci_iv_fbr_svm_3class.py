import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from filterbank import FilterBankRiemannian

DATA_DIR    = os.path.join("..", "data", "clean_bci_iv_3class")
RESULTS_DIR = os.path.join("results", "BCI_FBR_SVM_3class")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["Left Hand", "Right Hand", "Feet"]

def train_bci_iv_3class():
    file_list = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith("subject_") and "T_clean-epo.fif" in f
    ])

    if not file_list:
        print("Errore: Nessun file trovato.")
        return

    accuracies  = []
    all_y_true  = []
    all_y_pred  = []
    results_log = []

    print("Inizio training FBR+SVM 3 classi su BCI IV 2a...")
    print("--" * 30)

    for train_file in file_list:
        subject_id = train_file.split("_")[1]

        test_file  = train_file.replace("T_clean-epo.fif", "E_clean-epo.fif")
        train_path = os.path.join(DATA_DIR, train_file)
        test_path  = os.path.join(DATA_DIR, test_file)

        if not os.path.exists(test_path):
            print(f"Sessione E mancante per {subject_id}!")
            continue

        epochs_train = mne.read_epochs(train_path, verbose=False)
        epochs_test  = mne.read_epochs(test_path,  verbose=False)

        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:, -1]
        X_test  = epochs_test.get_data()
        y_test  = epochs_test.events[:, -1]

        # Pipeline identica alla versione binaria
        pipeline = Pipeline([
            ('FB',     FilterBankRiemannian(sfreq=250)),
            ('Scaler', StandardScaler()),
            ('CLF',    SVC(kernel='rbf'))
        ])

        param_grid = {
            'FB__estimator': ['lwf', 'oas'],
            'CLF__C':        [0.1, 1, 10, 100],
            'CLF__gamma':    ['scale', 'auto'],
            'CLF__decision_function_shape': ['ovr', 'ovo']
        }

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1
        )

        grid.fit(X_train, y_train)

        y_pred   = grid.predict(X_test)
        test_acc = np.mean(y_pred == y_test)

        accuracies.append(test_acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        results_log.append({
            'subject':     subject_id,
            'accuracy':    round(test_acc, 4),
            'best_params': grid.best_params_
        })

        print(f"Soggetto {subject_id} | Accuracy: {test_acc:.2%} | "
              f"Best params: {grid.best_params_}")

    if len(accuracies) == 0:
        print("Nessun soggetto processato!")
        return

    print("=" * 30)
    print(f"MEDIA TOTALE FBR+SVM 3 classi BCI IV: {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    np.save(os.path.join(RESULTS_DIR, "accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)

    # Confusion matrix 3x3 normalizzata
    cm     = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    # Valori numerici in ogni cella
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    ax.set_title("Confusion Matrix — BCI IV 2a FBR+SVM (3 classi)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    train_bci_iv_3class()


# ACCURACY 75.45%, MINIMA: 55.92% | MASSIMA: 96.19%
#MEDIA TOTALE FBR+SVM 3 classi BCI IV: 75.79%
#Minima: 53.08% | Massima: 96.67%