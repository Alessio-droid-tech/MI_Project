# PARAMETRI GLOBALI
SFREQ = 250
N_EEG_CHANNELS = 22
N_EOG_CHANNELS = 3


# Event Mapping basato sulla documentazione del pdf
EVENT_MAPPING = {
    769: 0,
    770: 1
}

# Eventi da ignorare (artefatti)
ARTIFACT_EVENT = 1023

MI_CLASSES = {0: "Left Hand", 1: "Right Hand"}

# EPOCHING (basato sul pdf)
# La task inizia a 2.0 secondi e finisce a 6.0 secondi
EPOCH_TMIN = 2.5    # Scarto di mezzo secondo per reaction time
EPOCH_TMAX = 5.5

# LIsta dei 22 canali EEG utilizzati dal dataset
EEG_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
    'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]


# Canali EOG da rimuovere pre CSP
EOG_CHANNELS = ['EOG-left', 'EOG-central', 'EOG-right']