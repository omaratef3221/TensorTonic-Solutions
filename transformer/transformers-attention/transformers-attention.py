import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """

    K_transposed = torch.transpose(K, -1, -2)

    numerator = torch.matmul(Q, K_transposed)
    den = torch.sqrt(torch.tensor(K.shape[-1]))
    inside_term = torch.div(numerator, den)
    softmax_term = torch.nn.functional.softmax(inside_term, dim = -1)
    final_output = torch.matmul(softmax_term , V.float())

    return final_output