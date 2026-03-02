import mne
from config import EPOCH_TMIN, EPOCH_TMAX, MI_CLASSES


def create_epochs(raw, events):
    """
    Crea epoch dal segnale EEG basato sugli eventi forniti.
    """
    # FIX: MNE vuole dizionario con formato {'Nome': ID}
    event_id = {v: k for k, v in MI_CLASSES.items()}  # Dizionario di mapping delle classi
    

    # Creazione di oggetti Epochs in MNE -> Segmenti di dati EEG tagliati attorno a determinati eventi.
    epochs = mne.Epochs(
        raw,                # Oggetto Raw con i dati continui EEG caricati
        events,             # Matrice degli eventi
        event_id=event_id,  # Dizionario {Nome: ID} per selezionare quali eventi estrarre dal raw
        tmin=EPOCH_TMIN,    # Tempo di inizio dell'epoca rispetto all'evento (in secondi)
        tmax=EPOCH_TMAX,    # Tempo di fine dell'epoca rispetto all'evento (in secondi)
        baseline=None,      # Nessuna correzione di baseline (per applicare CSP dopo nel ML)
        preload=True,       # Carica gli epoch in memoria (RAM)
        verbose=False,      # Disattiva messaggi di log dettagliati per non avere output troppo lungo
        on_missing='warn'   # Se un evento non viene trovato, mostra un warning invece di fermare il programma
    )

    return epochs