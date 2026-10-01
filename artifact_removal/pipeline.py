from data_loader import load_sig_csv, load_ann_csv
from preprocessing import apply_filters
from artifact_removal import remove_artifacts
from epoching import create_epochs

def process_run(sig_path, ann_path):
    # Caricamento e Filtri
    raw = load_sig_csv(sig_path) # Caricamento del file SIG
    raw = apply_filters(raw) # Applicazione dei filtri

    # Pulizia (ICA)
    raw_clean, labels = remove_artifacts(raw) # Rimozione degli artefatti

    # Caricamento Eventi
    events = load_ann_csv(ann_path)
   
    # Creazione Epoche (ritorna oggetto MNE! non npy)
    epochs = create_epochs(raw_clean, events)

    # Ritorna oggetto epochs (contenente dati ed etichette)
    return epochs

if __name__ == "__main__":
    pass
