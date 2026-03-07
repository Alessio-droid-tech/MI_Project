import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix

DATA_DIR    = os.path.join("..", "data", "clean")
RESULTS_DIR = os.path.join("results", "PHY_CSP_LDA")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_SIZE    = 0.2
RANDOM_STATE = 42

def train_physionet_csp_lda():
    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("-epo.fif")])

    if not file_list:
        print(f"Errore: Nessun file trovato in {os.path.abspath(DATA_DIR)}!")
        return

    print(f"Trovati {len(file_list)} soggetti. Inizio CSP+LDA su PhysioNet...")
    print("--" * 30)

    accuracies  = []
    all_y_true  = []
    all_y_pred  = []
    results_log = []

    for filename in file_list:
        subject_id = filename.split("_")[1]
        epochs = mne.read_epochs(os.path.join(DATA_DIR, filename), verbose=False)

        epochs.filter(l_freq=8.0, h_freq=30.0, verbose=False)
        epochs.crop(tmin=0.5, tmax=3.5)

        X = epochs.get_data()
        y = epochs.events[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )

        pipeline = Pipeline([
            ('CSP', CSP(reg='ledoit_wolf', log=True, norm_trace=False)),
            ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        param_grid = {'CSP__n_components': [4, 6, 8]}

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

    print("=" * 30)
    print(f"MEDIA TOTALE CSP+LDA PhysioNet: {np.mean(accuracies):.2%}")
    print(f"Minima: {np.min(accuracies):.2%} | Massima: {np.max(accuracies):.2%}")

    np.save(os.path.join(RESULTS_DIR, "phy_csp_lda_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "phy_csp_lda_results_log.json"), 'w') as f:
        json.dump(results_log, f, indent=2)

    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - PhysioNet CSP+LDA (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "phy_csp_lda_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    train_physionet_csp_lda()