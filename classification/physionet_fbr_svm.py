import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from filterbank import FilterBankRiemannian

MOTOR_CHANNELS = [
    'C3', 'C4', 'Cz',
    'FC3', 'FC4', 'FCz',
    'CP3', 'CP4', 'CPz',
    'C1', 'C2', 'C6'
]

DATA_DIR    = os.path.join("..", "data", "clean")
RESULTS_DIR = os.path.join("results", "PHY_FBR_SVM")
os.makedirs(RESULTS_DIR, exist_ok=True)

SFREQ        = 160   # PhysioNet usa 160 Hz, non 250
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def train_physionet_fbr_svm():
    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("-epo.fif")])

    if not file_list:
        print(f"Errore: Nessun file trovato in {os.path.abspath(DATA_DIR)}!")
        return

    print(f"Trovati {len(file_list)} soggetti. Inizio FBR+SVM su PhysioNet...")
    print("--" * 30)

    accuracies  = []
    all_y_true  = []
    all_y_pred  = []
    results_log = []

    # Parametri fissi: i migliori osservati su BCI IV
    # sfreq=160 fondamentale: i filtri Butterworth usano la fs
    # per calcolare le frequenze di taglio — sbagliarlo invalida tutto
    pipeline = Pipeline([
        ('FB',     FilterBankRiemannian(sfreq=SFREQ, estimator='lwf')),
        ('Scaler', StandardScaler()),
        ('PCA', PCA(n_components=33)),
        ('CLF',    SVC(kernel='rbf', C=0.1, gamma='scale'))
    ])

    for filename in file_list:
        subject_id = filename.split("_")[1]
        epochs = mne.read_epochs(os.path.join(DATA_DIR, filename), verbose=False)

        # Crop mantenuto: rimuove artefatti post-cue tipici di PhysioNet
        # Nessun filtro 8-30 Hz: lo fa FilterBankRiemannian internamente
        epochs.crop(tmin=0.5, tmax=3.5)

        available = [ch for ch in MOTOR_CHANNELS if ch in epochs.ch_names]
        epochs.pick_channels(available)

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
    print(f"MEDIA TOTALE FBR+SVM PhysioNet: {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    np.save(os.path.join(RESULTS_DIR, "phy_fbr_svm_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "phy_fbr_svm_results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)

    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - PhysioNet FBR+SVM (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "phy_fbr_svm_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    train_physionet_fbr_svm()