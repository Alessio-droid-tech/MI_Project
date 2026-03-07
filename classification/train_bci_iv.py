import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from filterbank import FilterBankRiemannian

# --- CONFIGURAZIONE ---
DATA_DIR = os.path.join("..", "data", "clean_bci_iv") # La nuova cartella
RESULTS_DIR = os.path.join("results", "BCI_FBR_SVM")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_bci_iv():
    file_list = sorted([f for f in os.listdir(DATA_DIR) 
                        if f.startswith("subject_") and "T_clean-epo.fif" in f])
    
    if not file_list:
        print("Errore: Nessun file trovato.")
        return

    accuracies = list()

    # Liste per confusion matrix
    all_y_true = list()
    all_y_pred = list()
    results_log = list() # Per salvarei best params per soggetto

    print("Inizio training su dataset BCI IV (Soggetti addestrati)...")
    print("--" * 30)

    for train_file in file_list:
        subject_id = train_file.split("_")[1]   # Es: A01T
        #file_path = os.path.join(DATA_DIR, filename)

        test_file = train_file.replace("T_clean-epo.fif", "E_clean-epo.fif")

        train_path = os.path.join(DATA_DIR, train_file)
        test_path = os.path.join(DATA_DIR, test_file)

        if not os.path.exists(test_path):
            print(f"Sessione E mancante per {subject_id}!")
            continue

        # Caricamento dei dati
        epochs_train = mne.read_epochs(train_path, verbose=False)
        epochs_test = mne.read_epochs(test_path, verbose=False)

        # Niente filtro 8-30 Hz perchè lo fa FilterBank
        epochs_train = mne.read_epochs(train_path, verbose=False)
        epochs_test = mne.read_epochs(test_path, verbose=False)

        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:, -1]

        X_test = epochs_test.get_data()
        y_test = epochs_test.events[:, -1]
        
        # Pipeline usando Filter Bank + StandardScaler + SVM
        pipeline = Pipeline([
            ('FB', FilterBankRiemannian(sfreq=250)),
            ('Scaler', StandardScaler()),
            ('CLF', SVC(kernel='rbf'))
        ])

        param_grid = {
            'FB__estimator': ['lwf', 'oas'],
            'CLF__C': [0.1, 1, 10],
            'CLF__gamma': ['scale', 'auto']
        }

        # Grid Search
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,               # cv solo sulla sessione T
            scoring='accuracy',
            n_jobs=1
        )

        # TRAIN su T
        grid.fit(X_train, y_train)

        # TEST su E
        y_pred = grid.predict(X_test)
        test_acc = np.mean(y_pred == y_test)

        accuracies.append(test_acc)

        # Salvataggio globale per la confusion matrix
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Salvataggio risultati per soggetto
        results_log.append({
            'subject': subject_id,
            'accuracy': round(test_acc, 4),
            'best_params': grid.best_params_
        })

        print(f"Soggetto {subject_id} | Accuracy: {test_acc:.2%}")
        print(f"Best params (da T): {grid.best_params_}")

    # Check di sicurezza per evitare crash
    if len(accuracies) == 0:
        print("Nessun soggetto processato causa errori!")
        return
    

    print("=" * 20)
    print(f"MEDIA TOTALE BCI IV 2a (TRAIN T -> TEST E): {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    # Salvo risultati per grafici + grafici comparativi
    np.save(os.path.join(RESULTS_DIR, "bci_iv_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "BCI_Competition_results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)


    # GRAFICO CONFUSION MATRIX GLOBALE
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Normalizzazione per riga:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    # Aggiunta valori numerici in ogni cella
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight='bold')

    ax.set_title("Confusion Matrix - BCI Competition IV 2a")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Left Hand", "Right Hand"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "physionet_csp_svm_confusion_matrix.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    train_bci_iv()