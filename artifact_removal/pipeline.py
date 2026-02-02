from data_loader import load_sig_csv, load_ann_csv
from preprocessing import apply_filters
from artifact_removal import remove_artifacts
from epoching import create_epochs

def process_run(sig_path, ann_path):
    raw = load_sig_csv(sig_path) # Caricamento del file SIG
    raw = apply_filters(raw) # Applicazione dei filtri

    raw_clean, labels = remove_artifacts(raw) # Rimozione degli artefatti

    events = load_ann_csv(ann_path) # Caricamento del file ANN
    epochs = create_epochs(raw_clean, events) # Creazione degli epoch

    x = epochs.get_data() # Estrazione dei dati dagli epoch
    y = epochs.events[:, -1] # Estrazione delle etichette degli eventi

    return x, y

if __name__ == "__main__":
    x, y = process_run(
        "SUB_01_SIG_01.csv", 
        "SUB_01_ANN_01.csv"
    )
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)

