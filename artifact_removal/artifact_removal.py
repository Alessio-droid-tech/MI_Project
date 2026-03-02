# artifact_removal.py
from mne.preprocessing import ICA
from mne.io import BaseRaw
from mne_icalabel import label_components
from config import ICA_COMPONENTS, RANDOM_STATE

def remove_artifacts(raw: BaseRaw):
    """
    Rimozione AUTOMATICA degli artefatti EEG usando ICA + ICLabel.
    Questa versione serve per soddisfare i requisiti della IA di ICLabel (CAR -> Common Average Reference e 1-100Hz)
    """
    # Creazione di una copia solo per il calcolo dell'ICA
    raw_for_ica = raw.copy()

    raw_for_ica.set_eeg_reference('average', projection=False, verbose=False)

    # Filtro HIGH-PASS robusto per il fitting dell'ICA. ( e modificato 1-100 Hz per ICLabel)
    raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False) # Con Higher = 100 errore perché Hz max = 80 (limite creato da Nyquist)

    # Fit dell'ICA utilizzando algoritmo picard (o fastICA)
    # Modalità EXTENDED per avere più possibilità di individuare un artefatto
    ica = ICA(
        n_components=ICA_COMPONENTS,  # Numero di componenti indipendenti da estrarre (definito in config.py)
        method="picard",              # Algoritmo di ottimizzazione utilizzato
        fit_params=dict(
            ortho=False,              # Disabilità il vincolo di ortogonalità fra le componenti
            extended=True             # Abilita modalità extended-ICA (per separazione sorgenti sub-Gaussiane da super-Gaussiane)
        ),
        random_state=RANDOM_STATE,    # Seed per la riproducibilità dei risultati (definito in config.py)
        max_iter="auto",              # Numero massimo di iterazioni (con automatico si adatta alla convergenza)
        verbose=False                 # Disattiva messaggi di log dettagliati per non avere output troppo lungo
    )
    ica.fit(raw_for_ica)


    # Classificazione AUTOMATICA con ICLabel (ML)
    # Ritorna le etichette:
    # 'brain', 'muscle', 'eye', 'heart', 'line_noise', 'other'
    component_labels = label_components(raw_for_ica, ica, method='iclabel')

    exclude_idx = list()
    labels = component_labels["labels"]
    probs = component_labels["y_pred_proba"]


    # Settore per esclusione:
    # rimuoviamo tutto ciò che è classificato, con alta probabilità, come artefatto.
    print(" -> Analisi componenti ICA:")
    for i, label in enumerate(labels):
        prob = probs[i]

         # Per DEBUG
        print(f"     Comp {i:02d}: {label.upper():<12} (Prob: {prob:.1%})", end="")

        # Logica per esclusione
        # if label in ['muscle', 'eye', 'heart', 'line_noise', 'channel_noise']:
        # Inversione dell'if per etichette che non trovava

        is_artifact = label not in ['brain', 'other']
        
        if is_artifact and prob > 0.30: # Abbassamento soglia a 30% per vedere se ICA detecta qualcosa (bassa, cercare di alzarla)
            # Rimozione artefatto
            exclude_idx.append(i)
            print(f" [X] Comp {i}: {label.upper()} ({prob:.2%}) => RIMOSSA!")
        else:
            print(f" [ ] Comp {i}: {label} ({prob:.2%}) => MANTENUTA!")
    
    ica.exclude = exclude_idx

    # Print per DEBUG:
    print(f" -> ICA ha rimosso {len(exclude_idx)} componenti: {exclude_idx}")


    # Applica la pulizia elaborata al seganle originale
    raw_clean = ica.apply(raw.copy(), verbose=False)

    return raw_clean, labels