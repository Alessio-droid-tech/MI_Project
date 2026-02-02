from config import LOW_FREQ, HIGH_FREQ, NOTCH_FREQ # Added import statements

def apply_filters(raw):
    """
    Applica filtri passa-banda e notch al segnale EEG.
    """
    # Filtro passa-banda
    raw.filter(
        l_freq=LOW_FREQ, 
        h_freq=HIGH_FREQ, 
        fir_design='firwin'
    )
    
    # Filtro notch
    raw.notch_filter(
        freqs=NOTCH_FREQ
    )
    
    return raw