import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from data_loader import load_sig_csv
from config import ICA_COMPONENTS, RANDOM_STATE

# Usare lo stesso file csv specificato in check_cleaning per chiarezza
#TEST_FILE = "../eegmmidb/SUB_001_SIG_02.csv" # Tre artefatti
TEST_FILE = "../eegmmidb/SUB_063_SIG_06.csv" # Sedici artefatti

def inspect():
    print(f"Caricamento file {TEST_FILE}...")

    # Caricamento e preprocessing
    raw = load_sig_csv(TEST_FILE)

    raw_for_ica = raw.copy()
    raw_for_ica.set_eeg_reference('average', projection=False, verbose=False)
    raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)


    # Calcolo ICA con PICARD
    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=RANDOM_STATE,
        max_iter="auto",
        verbose=False
    )
    ica.fit(raw_for_ica)

    
    # Classificazione AUTOMATICA con ICLabel (ML)
    component_labels = label_components(raw_for_ica, ica, method='iclabel')

    exclude_idx = list()
    labels = component_labels["labels"]
    probs = component_labels["y_pred_proba"]


    # Settore per esclusione:
    # rimuoviamo tutto ciò che è classificato, con alta probabilità, come artefatto.
    print("--- COMPONENTI IDENTIFICATE DALL'IA ---")
    for i, label in enumerate(labels):
        prob = probs[i]

        is_artifact = label not in ['brain', 'other']
        
        if is_artifact and prob > 0.30: # Abbassamento soglia a 30% per vedere se ICA detecta qualcosa (bassa, cercare di alzarla)
            # Rimozione artefatto
            exclude_idx.append(i)
            print(f" -> Comp {i:02d}: {label.upper()} ({prob:.1%}) [DA RIMUOVERE]")
        else:
            print(f" -> Comp {i:02d}: {label} [OK]!")
            pass
    
    print(f"Visualizzazione grafici per le componenti: {exclude_idx}")


    # PLOT 1: Topomaps
    # Visualizzazione su dove sono localizzate le componenti
    ica.plot_components(picks=exclude_idx, title="Componenti Artefatto (Topomaps)")
    # ATTUALMENTE NON FUNZIONANTE -> plt.savefig("../images/topomaps.pdf", format = "pdf", bbox_inches= "tight")

    # PLOT 2: Sources (onde)
    # Variazione nel tempo
    ica.plot_sources(raw_for_ica, picks=exclude_idx, title="Componenti Artefatto", show_scrollbars=False)

    # PLOT 3: Overlays (visualizzazione Prima vs Dopo)
    # Effetti su segnale togliendo artefatto
    ica.exclude = exclude_idx # Qui fai esclusione
    ica.plot_overlay(raw_for_ica, exclude=exclude_idx, picks="eeg", title="Effetto Pulizia (Rosso = Prima, Nero = Dopo)")

    # PLOT 4: Component properties
    ica.plot_properties(raw, picks=[0])

    # PLOT 5: Power Spectral Density (PSD)
    raw.plot_psd()
    raw_for_ica.plot_psd()

    plt.show()

if __name__ == "__main__":
    inspect()    