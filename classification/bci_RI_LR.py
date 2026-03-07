import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# --- CONFIGURAZIONE ---
DATA_DIR    = os.path.join("..", "data", "clean_bci_iv")
RESULTS_DIR = os.path.join("results", "BCI_RI_LR")
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_bci_iv_riemannian_lr():
    file_list = sorted([f for f in os.listdir(DATA_DIR)
                        if f.startswith("subject_") and "T_clean-epo.fif" in f])

    if not file_list:
        print("Errore: Nessun file trovato.")
        return

    accuracies  = list()
    all_y_true  = list()
    all_y_pred  = list()
    results_log = list()

    print("Inizio training Riemannian + Logistic Regression su dataset BCI IV...")
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

        # Nessun filtro 8-30 Hz: il segnale broadband va bene per le
        # matrici di covarianza. Il filtraggio spettrale non è necessario
        # come lo è per CSP, che dipende dalla potenza di banda.
        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:, -1]
        X_test  = epochs_test.get_data()
        y_test  = epochs_test.events[:, -1]

        # Pipeline Riemanniana base:
        # Covariances -> TangentSpace -> StandardScaler -> LogisticRegression
        pipeline = Pipeline([
            ('Cov',    Covariances()),
            ('TS',     TangentSpace()),
            ('Scaler', StandardScaler()),
            ('LR',     LogisticRegression(max_iter=1000, solver='lbfgs'))
        ])

        param_grid = {
            'Cov__estimator': ['lwf', 'oas'],
            'LR__C':          [0.1, 1, 10]
        }

        grid = GridSearchCV(pipeline, param_grid, cv=5,
                            scoring='accuracy', n_jobs=1)

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

    print("=" * 20)
    print(f"MEDIA TOTALE BCI IV 2a Riemannian+LR: {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    np.save(os.path.join(RESULTS_DIR, "bci_RI_LR_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "bci_RI_LR_results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)

    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - BCI IV 2a Riemannian+LR (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bci_ri_lr_confusion_matrix.png"))
    plt.show()


if __name__ == "__main__":
    train_bci_iv_riemannian_lr()