import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

DATA_DIR    = os.path.join("..", "data", "clean")
RESULTS_DIR = os.path.join("results", "PHY_RI_LR")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_SIZE    = 0.2
RANDOM_STATE = 42

def train_physionet_riemannian_lr():
    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("-epo.fif")])

    if not file_list:
        print(f"Errore: Nessun file trovato in {os.path.abspath(DATA_DIR)}!")
        return

    print(f"Trovati {len(file_list)} soggetti. Inizio Riemannian+LR su PhysioNet...")
    print("--" * 30)

    accuracies  = []
    all_y_true  = []
    all_y_pred  = []
    results_log = []

    # Parametri fissi: i migliori osservati su BCI IV
    # Nessuna GridSearchCV -> drastica riduzione dei tempi
    pipeline = Pipeline([
        ('Cov',    Covariances(estimator='lwf')),
        ('TS',     TangentSpace()),
        ('Scaler', StandardScaler()),
        ('LR',     LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
    ])

    for filename in file_list:
        subject_id = filename.split("_")[1]
        epochs = mne.read_epochs(os.path.join(DATA_DIR, filename), verbose=False)

        # Nessun filtro 8-30 Hz: le matrici di covarianza funzionano
        # meglio sul segnale broadband
        epochs.crop(tmin=0.5, tmax=3.5)

        X = epochs.get_data()
        y = epochs.events[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )

        pipeline.fit(X_train, y_train)
        y_pred   = pipeline.predict(X_test)
        test_acc = np.mean(y_pred == y_test)

        accuracies.append(test_acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        results_log.append({'subject': subject_id, 'accuracy': round(test_acc, 4)})

        print(f"Soggetto {subject_id} | Accuracy: {test_acc:.2%}")

    print("=" * 30)
    print(f"MEDIA TOTALE Riemannian+LR PhysioNet: {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    np.save(os.path.join(RESULTS_DIR, "phy_ri_lr_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "phy_ri_lr_results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)

    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - PhysioNet Riemannian+LR (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "phy_ri_lr_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    train_physionet_riemannian_lr()