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
    raw_for_ica.filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # Fit dell'ICA utilizzando algoritmo picard (o fastica)
    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="picard",
        random_state=RANDOM_STATE,
        max_iter="auto",
        verbose=False
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

        # Logica per esclusione
        if label in ['muscle', 'eye', 'heart', 'line_noise', 'channel_noise']:
            if prob > 0.70:
                # Rimozione artefatto
                exclude_idx.append(i)
                print(f" [X] Comp {i}: {label.upper()} ({prob:.2%}) => RIMOSSA!")
            else:
                print(f" [ ] Comp {i}: {label} ({prob:.2%}) => DUBBIO (MANTENUTA)!")
        else:
            # Brain o Other
            pass
    
    ica.exclude = exclude_idx

    # Print per DEBUG:
    print(f" -> ICA ha rimosso {len(exclude_idx)} componenti: {exclude_idx}")


    # Applica la pulizia elaborata al seganle originale
    raw_clean = ica.apply(raw.copy(), verbose=False)

    return raw_clean, labels