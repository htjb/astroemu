import torch

def compute_mean_std(loader: torch.utils.data.DataLoader) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ):
    """
    Memory safe mean and std computation
    
    Args:
        loader: DataLoader returning (spec, input) where:
            - spec: [batch_size, 5000] 
            - input: [batch_size, 5000, N]
    
    Returns:
        mean_spec: [5000] - mean across batches
        std_spec: [5000] - std across batches  
        mean_input: [N] - mean across batches and the 5000 dimension
        std_input: [N] - std across batches and the 5000 dimension
    """
    # Accumulators
    spec_sum = None
    spec_sum_sq = None
    input_sum = None  
    input_sum_sq = None
    n_spec_samples = 0
    n_input_samples = 0
    
    for spec, input_data in loader:
        batch_size = spec.size(0)
        
        # === Process spec ===
        # spec shape: [batch_size, 5000] -> we want stats across batch dim
        if spec_sum is None:
            spec_sum = torch.zeros(spec.size(1), dtype=spec.dtype, device=spec.device)
            spec_sum_sq = torch.zeros(spec.size(1), dtype=spec.dtype, device=spec.device)
        
        spec_sum += spec.sum(dim=0)  # sum across batch
        spec_sum_sq += (spec ** 2).sum(dim=0)  # sum of squares across batch
        n_spec_samples += batch_size
        
        # === Process input ===  
        # input shape: [batch_size, 5000, N] -> we want stats across batch and 5000 dims
        input_flat = input_data.view(-1, input_data.size(-1))  # [batch_size * 5000, N]
        
        if input_sum is None:
            input_sum = torch.zeros(input_data.size(-1), dtype=input_data.dtype, device=input_data.device)
            input_sum_sq = torch.zeros(input_data.size(-1), dtype=input_data.dtype, device=input_data.device)
        
        input_sum += input_flat.sum(dim=0)  # sum across flattened batch*5000 dim
        input_sum_sq += (input_flat ** 2).sum(dim=0)  # sum of squares
        n_input_samples += input_flat.size(0)  # batch_size * 5000
    
    # Compute means and stds
    mean_spec = spec_sum / n_spec_samples
    var_spec = (spec_sum_sq / n_spec_samples) - (mean_spec ** 2)
    std_spec = torch.where(var_spec <= 1e-3, 1, torch.sqrt(var_spec))
    
    mean_input = input_sum / n_input_samples  
    var_input = (input_sum_sq / n_input_samples) - (mean_input ** 2)
    std_input = torch.sqrt(torch.clamp(var_input, min=1e-8))  # clamp for numerical stability
    
    return mean_spec, std_spec, mean_input, std_input