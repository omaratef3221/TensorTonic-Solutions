import numpy as np
from scipy.special import softmax

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    eps = 1e-6
    mean = x.mean(axis = -1, keepdims = True)
    sigma = x.var(axis = -1, keepdims = True)
    
    numerator = x - mean
    den = np.sqrt(sigma + eps)
    
    division_result = numerator/den
    
    result = gamma * division_result + beta
    
    return result

def multi_head_attention(Q,K,V, W_q, W_k, W_v, W_o, num_heads):
    ### MHA
    batch_size, seq_len_q, d_model = K.shape[0], Q.shape[1], K.shape[2]
    seq_len_k = K.shape[1]
    seq_len_v = V.shape[1]
    
    d_k = d_model // num_heads
    
    Q_ = Q @ W_q # ---> (Batch Size, seq_len, d_model)
    K_ = K @ W_k # ---> (Batch Size, seq_len, d_model)
    V_ = V @ W_v # ---> (Batch Size, seq_len, d_model)

    Q_ = Q_.reshape(batch_size, seq_len_q, num_heads, d_k)
    K_ = K_.reshape(batch_size, seq_len_k, num_heads, d_k)
    V_ = V_.reshape(batch_size, seq_len_v, num_heads, d_k)
    
    Q_ = np.transpose(Q_, axes = [0, 2, 1, 3])
    K_ = np.transpose(K_, axes = [0, 2, 3, 1])
    V_ = np.transpose(V_, axes = [0, 2, 1, 3]) 
    # ---> Q, V(Batch Size, num_heads, seq_len, d_k)
    # ---> K (Batch Size, num_heads, d_k, seq_len)
    
    head_i = softmax((Q_ @ K_) / np.sqrt(d_k), axis = -1) @ V_
    head_i = np.transpose(head_i, axes = [0, 2, 1, 3])
    head_i = head_i.reshape(batch_size, seq_len_v, d_model)

    attn_result = head_i @ W_o
    return attn_result

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.maximum(0, x@W1 + b1)@W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    ### MHA
    attn_result = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    ### Norm 1
    z = layer_norm(x+attn_result, gamma1, beta1)

    ### FFN
    ffnz = feed_forward(z, W1, b1, W2, b2)

    ### Norm 2
    y = layer_norm(z+ffnz, gamma2, beta2)
    
    return y