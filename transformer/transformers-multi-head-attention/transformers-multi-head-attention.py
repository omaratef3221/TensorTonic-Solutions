import numpy as np
from scipy.special import softmax


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len, d_model = Q.shape[0], Q.shape[1], Q.shape[2]
    d_k = d_model // num_heads

    Q_ = Q @ W_q # ---> (Batch Size, seq_len, d_model)
    K_ = K @ W_k # ---> (Batch Size, seq_len, d_model)
    V_ = V @ W_v # ---> (Batch Size, seq_len, d_model)


    Q_ = Q_.reshape(batch_size, seq_len, num_heads, d_k) 
    Q_transpose = np.transpose(Q_, axes = [0, 2, 1, 3]) # --> (batch_size, num_heads, seq_len, d_k) 

    V_ = V_.reshape(batch_size, seq_len, num_heads, d_k)
    V_transpose = np.transpose(V_, axes = [0, 2, 1, 3]) # --> (batch_size, num_heads, seq_len, d_k) 

    K_ = K_.reshape(batch_size, seq_len, num_heads, d_k)
    K_transposed = np.transpose(K_, axes = [0, 2, 3, 1]) # --> (batch_size, num_heads, d_k, seq_len) 


    numerator= np.matmul(Q_transpose, K_transposed)
    den = np.sqrt(d_k)

    inside_term = np.divide(numerator, den)
    softmax_term = softmax(inside_term, axis = -1)
    final_head_i = np.matmul(softmax_term, V_transpose)


    final_head_i = np.transpose(final_head_i, axes = [0, 2, 1, 3])
    final_head_i = final_head_i.reshape(batch_size, seq_len, num_heads*d_k)

    final_output = final_head_i @ W_o

    return final_output