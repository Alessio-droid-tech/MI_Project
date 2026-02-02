# PARAMETRI GLOBALI (costanti)
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 60.0 # Based on american Hz value.
SFREQ = 160
N_CHANNELS = 64

# SELEZIONE DEI DATI basata sulla tabella 2 a https://www.sciencedirect.com/science/article/pii/S2352340924001525?utm_source=chatgpt.com#fig0001
# Motor Imagery per Left/Right fist corrisponde ai file 2 6 10. Se si vuole implementare anche i Both Feet bisogna usare 4 8 e 12. Se si vuole implementare anche il Motor Execution, tutti gli altri.
TARGET_RUNS = [2, 6, 10] # Questi per addrestrare un modello Left vs Right

# Il mapping segue sempre la tabella (ultima colonna)
EVENT_MAPPING = {
    5: 0, # Classifico Left come classe 0 per la AI
    6: 1 # E Right come classe 1 per la AI
} # T0 sarebbe = 4 che equivale a RELAX. Modificare se si vuole implementare questa fase.


MI_CLASSES = {0: "Left Hand", 1: "Right Hand"}

# EPOCHING
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0

# PARAMETRI ICA
ICA_COMPONENTS = 20
RANDOM_STATE = 97



# LISTA DEI CANALI usando lo standard 10-10 usato da PhysioNet
CH_NAMES = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz',
    'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7',
    'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
] # Nomi in maiuscolo perché lo standard_1005 di MNE si aspetta che i nomi dei canali combinati siano maiuscolo. (Fp viene accettato minuscolo)