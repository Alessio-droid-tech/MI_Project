import mne
from config import EPOCH_TMIN, EPOCH_TMAX, MI_CLASSES


def create_epochs(raw, events):
    """
    Crea epoch dal segnale EEG basato sugli eventi forniti.
    """
    event_id = MI_CLASSES  # Dizionario di mapping delle classi
    
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id,
        tmin=EPOCH_TMIN, 
        tmax=EPOCH_TMAX,
        baseline=None,  # Nessuna correzione di baseline
        preload=True    # Carica gli epoch in memoria
    )

    return epochs