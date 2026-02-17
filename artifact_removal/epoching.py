import mne
from config import EPOCH_TMIN, EPOCH_TMAX, MI_CLASSES


def create_epochs(raw, events):
    """
    Crea epoch dal segnale EEG basato sugli eventi forniti.
    """
    # FIX: MNE vuole dizionario con formato {'Nome': ID}
    event_id = {v: k for k, v in MI_CLASSES.items()}  # Dizionario di mapping delle classi
    
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id,
        tmin=EPOCH_TMIN, 
        tmax=EPOCH_TMAX,
        baseline=None,  # Nessuna correzione di baseline (se si applica CSP dopo in classificazione MI)
        preload=True,   # Carica gli epoch in memoria
        verbose=False,
        on_missing='warn' # Per evitare crash
    )

    return epochs