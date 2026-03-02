import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sb
#from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix

# Pacchetti per modifica da CSP a Riemannian Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from filterbank import FilterBankRiemannian

# --- CONFIGURAZIONE ---
DATA_DIR = os.path.join("..", "data", "clean_bci_iv") # La nuova cartella
RESULTS_DIR = "results"
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
        

        # ====================================================================
        # Caricamento Dati
        #epochs = mne.read_epochs(file_path, verbose=False)

        # Filtraggio Frequenziale (8-30 Hz per Motor Imagery)
        # Isolamento delle bande Mu e Beta.
        #epochs.filter(l_freq=8.0, h_freq=30.0, verbose=False)

        #X = epochs.get_data()
        #y = epochs.events[:, -1]
        # ======= STRUTTURA PRE INSERIMENTO FILE E ========


        # ==========================================================================================
        # Pipeline di Machine Learning
        #csp = CSP(reg='ledoit_wolf', log=True, norm_trace=False) # No n_components per usare Grid Search
        #svm = SVC(kernel='linear', C=1)
        #svm = SVC(kernel='rbf', C=1, gamma='scale') POCO PIù BASSA DI KERNEL LINEAR
        #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        #pipeline = Pipeline([
        #    ('CSP', csp),
        #    ('SVM', svm)
        #])

        
        # Griglia dei parametri da testare (senza metterli fissi nel CSP per provare ad alzare accuracy)
        #param_grid = {
        #   'CSP__n_components': [4, 6, 8, 10],
        #   'SVM__C': [0.1, 1, 10]
        #}
        # ====== TUTTA COMMENTATA PER IMPLEMENTARE RIEMANNIAN PIPELINE PER AUMENTARE ACCURACY ======

        # Pipeline usando Riemannian
       # pipeline = Pipeline([
        #    ('Cov', Covariances()),                    # Calcola la matrice di covarianza su ogni trial usando Ledoit-Wolf Shrinkage Estimator
        ##    #('TS', TangentSpace()),                    # Mappa le matrici SPD su uno spazio vettoriale
        #    ('Scaler', StandardScaler()),              #
        #    ('LR', LogisticRegression(max_iter=1000))  # Classificazione
        #])

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

        # Griglia dei parametri da testare con Riemannian
        #param_grid = {
        #    'Cov__estimator': ['scm', 'lwf', 'oas'],
        #    'LR__C': [0.1, 1, 10]
        #}

        # Cross-Validation TOLTA PER USARE I FILE DI TIPO E
        #cv = ShuffleSplit(
        #    n_splits=N_SPLITS, 
        #    test_size=TEST_SIZE, 
        #    random_state=RANDOM_STATE
        #)
        # scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

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

        # GRAFICO CONFUSION MATRIX SU OGNI SOGGETTO
     #   cm = confusion_matrix(y_test, y_pred)

      #  plt.figure(figsize=(6,5))
      #  plt.imshow(cm, interpolation='nearest')
      #  plt.title(f"Confusion Matrix - {subject_id}")
      #  plt.colorbar()
      #  plt.xlabel("Predicted")
      #  plt.ylabel("True")
      #  plt.xticks(range(len(np.unique(y_test))))
      #  plt.yticks(range(len(np.unique(y_test))))
      #  plt.tight_layout()
      #  plt.show()

        # TEST su E
        #test_acc = grid.score(X_test, y_test)

        #mean_acc = grid.best_score_  # Prende il miglior risultato tra tutte le combinazioni provate della Grid Search
        #accuracies.append(test_acc)

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

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    #sb.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix - BCI IV 2a (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_bci_iv()