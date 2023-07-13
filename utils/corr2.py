# ---------------------- Correlation Coefficient -------------------------------
import torch

def corr2(x, y, batch_mode=False):
    """
    Calculate the mean correlation coefficient between single or batches of input data.
    
    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors with the same number of matrices and the same batch size if batch_mode is True 
        (dimensions should be (N, H, W) or (N, 1, H, W) for single and batch input).
    batch_mode : bool, optional
        Set True to calculate the correlation coefficient for batches of input data, by default False.
        
    Returns
    -------
    mean_corr_coeffs : float or torch.Tensor
        Mean correlation coefficient between the matrices in x and y for single input (float) 
        or mean correlation coefficients for each batch if batch_mode is True (torch.Tensor with shape: (B,)).
    """
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise ValueError("Both input tensors must be torch.Tensor objects.")
    
    if x.shape != y.shape:
        raise ValueError("Both input tensors must have the same dimensions.")
    
    # Remove singleton dimensions, if present
    x = torch.squeeze(x, dim=-3) if len(x.shape) == 4 else x
    y = torch.squeeze(y, dim=-3) if len(y.shape) == 4 else y

    if batch_mode:
        batch_size = x.shape[0]
        mean_corr_coeffs = torch.empty(batch_size, dtype=torch.float32)

        for batch_idx in range(batch_size):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]
            mean_corr_coeffs[batch_idx] = corr2(x_batch, y_batch)

        return mean_corr_coeffs

    else:
        x_mean_centered = x - x.mean(dim=(-2, -1)).reshape(-1, 1, 1)
        y_mean_centered = y - y.mean(dim=(-2, -1)).reshape(-1, 1, 1)

        r = (x_mean_centered * y_mean_centered).sum(dim=(-2, -1)) / torch.sqrt((x_mean_centered**2).sum(dim=(-2, -1)) * (y_mean_centered**2).sum(dim=(-2, -1)))
        mean_corr_coeff = r.mean().item()

        return mean_corr_coeff