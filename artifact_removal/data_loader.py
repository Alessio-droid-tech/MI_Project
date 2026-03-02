# Caricamento dei file .csv della forma SIG ed ANN
import numpy as np
import pandas as pd
import mne
from config import SFREQ, N_CHANNELS, CH_NAMES, EVENT_MAPPING

def load_sig_csv(path_sig):
    """
    Carica i file .csv di tipo SIG e lo converte in Raw MNE
    """
    df = pd.read_csv(path_sig, header=None)
    data = df.values.T
    data = data * 1e-6 # Trasformazione da uV a V (microvolt a volt)

    info = mne.create_info(
        ch_names=CH_NAMES, # Li prende dalla lista in config.py
        sfreq=SFREQ,
        ch_types=["eeg"] * N_CHANNELS
    )

    raw = mne.io.RawArray(data, info) # Creazione dell'oggetto Raw MNE
    raw.set_montage('standard_1005')
    return raw

def load_ann_csv(path_ann):
    """
    Carica le annotazioni e le mappa:
    Se trova 5 -> 0 (Per Left Hand)
    Se trova 6 -> 1 (Per Right Hand)
    Ignora 4 attualmente (=Rest) e tutti gli altri numeri [1..3] & [7..12] 
    poiché corrispondenti a task non analizzate nel nostro caso.
    """
    ann = pd.read_csv(path_ann, header=None)
    events = []

    for _, row in ann.iterrows():
        original_label = int(row[0]) # 4 5 o 6
        start_sample = int(row[3])

        if original_label in EVENT_MAPPING:
            # Convertiamo in etichetta per il ML (0 o 1)
            new_label = EVENT_MAPPING[original_label]
            events.append([start_sample, 0, new_label])
    
    return np.array(events, dtype=int)
