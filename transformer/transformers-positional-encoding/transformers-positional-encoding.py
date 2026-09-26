import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    output = np.zeros((seq_length, d_model))
    
    all_pos = np.arange(seq_length)[:, np.newaxis]
    all_i = np.arange(0, d_model, 2)

    div_term= 10000 ** (all_i/d_model)
    output[:,::2] = np.sin(all_pos / div_term)
    output[:,1::2] = np.cos(all_pos / div_term)

    return output