import os
import numpy as np
import mne
import json
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


DATA_DIR = os.path.join("..", "data", "clean")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Parametri per il Machine Learning (ShuffleSplit)
TEST_SIZE = 0.2 # 20% dati per testing, 80% per addestrare
RANDOM_STATE = 42 # Seme per riproducibilità

def train_physionet():
    """
    Addestra un modello specifico per ogni soggetto che compone il dataset Physionet.
    Usa una pipeline CSP + GridSearch
    """

    # Cerca i file .fif risultanti dal dataset pulito
    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("-epo.fif")])

    if not file_list:    # Check su recupero files
        print(f"ERRORE: Nessun file trovato in {os.path.abspath(DATA_DIR)}!")
        return
    else:
        print(f"Trovati {len(file_list)} soggetti differenti. Inizio della fase di training su dataset Physionet...")
        print("--" * 30)


    accuracies = list() # Lista vuota dove salvare i risultati di tutti i pazienti
    all_y_true = list()
    all_y_pred = list() # Per confusion matrix
    results_log = list() # Per salvare i risultati e confrontarli poi con dataset BCI Competition IV 2a

    for filename in file_list:
        subject_string = filename.split("_")
        subject_id = subject_string[1]       # Estrazione ID soggetto (es. "subject_001_clean-epo.fif")
        file_path = os.path.join(DATA_DIR, filename)


        # CARICAMENTO DEI DATI
        # Ottenuto tramite libreria mne che legge dati, etichette e nomi canali
        epochs = mne.read_epochs(file_path, verbose=False)

        # Filtraggio delle frequenze per isolare le bande utili al Motor Imagery ed escludere il rumore
        # Scelta fatta in seguito al primo run con i log: MEDIA TOTALE DATASET: 46.32%   Accuracy MINIMA = 22.22% | MASSIMA = 97.78%
        epochs.filter(l_freq=8.0, h_freq=30.0, verbose=False)

        # Filtraggio linea temporale: eslusione primi 0.5 secondi per riduzione rumore ed aume to accuracy
        epochs.crop(tmin=0.5, tmax=3.5) # Dei 4 secondi delle epoche che sono state create, manteniamo gli ultimi 3.5

        # Estrazione matrici per Scikit-learn
        X = epochs.get_data()      # (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]   # Etichette Letf e Right (0 ed 1)

        # Split train/test fisso - stesso approccio logico delle sessioni T/E ddel dataset BCI Competition IV 2a
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        # DEFINIZIONE DELLA PIPELINE
        # Utilizzo CSP (Common Spatial Pattern) -> per trovare i filtri spaziali migliori
        csp = CSP(reg='ledoit_wolf', log=True, norm_trace=False)

        # LDA -> Classificatore per decidere se è 0 o 1 (Left o Right)
        # CON LDA ACCURACY 55%
        # lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

        # Modifica in SVM poiché secondo molti paper offre accuracy nettamente migliori
        svm = SVC(kernel='rbf')

        # Creazione effettiva della pipeline con l'unione CSP + LDA
        pipeline = Pipeline([('CSP', csp), ('SVM', svm)], verbose=False)

        # GridSearch su n_components, C e gamma
        param_grid = {
            'CSP__n_components': [4, 6, 8],
            'SVM__C': [0.1, 1, 10],
            'SVM__gamma': ['scale', 'auto']
        }

        # CROSS-VALIDATION
        # Fa uno shuffle dei dati e divide in Train e Test i dati puliti per 5 volte
        #cv = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1
        )


        # Esecuzione del training e calcolo dell'accuratezza del modello
       # scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

        # Calcolo della media dei valori fra le 5 prove
        #mean_acc = np.mean(scores)

        # Fit 
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        test_acc = np.mean(y_pred == y_test)

        accuracies.append(test_acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        results_log.append({
            'subject': subject_id,
            'accuracy': round(test_acc, 4),
            'best_params': grid.best_params_
        })

        print(f"Soggetto {subject_id} | ACCURACY: {test_acc:.2%} | Best params: {grid.best_params_}")

    if len(accuracies) == 0:
        print("Nessun soggetto processato!")
        return

    # Logs per i risultati finali
    print("=" * 30)
    print(f"MEDIA TOTALE DATASET Physionet EEGMMIDB: {np.mean(accuracies):.2%}")
    print(f"MINIMA = {np.min(accuracies):.2%} | MASSIMA = {np.max(accuracies):.2%}")
    print("--" * 30)

    # Save dei dati per grafici con matplotlib
    np.save(os.path.join(RESULTS_DIR, "physioney_accuracies.npy"), accuracies)
    with open(os.path.join(RESULTS_DIR, "physionet_result_log.json"), mode='w') as f:
        json.dump(results_log, f, indent=2)
    print(f"Risultati salvati in {RESULTS_DIR}/physionet_accuracies.npy")


    # Confusion matrix globale normalizzata
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - PhysioNet EEGMMIDB (Global)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Left Hand", "Right Hand"])
    plt.yticks([0, 1], ["Left Hand", "Right Hand"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "physionet_confusion_matrix.png"))
    plt.show()


if __name__ == "__main__":
    train_physionet()