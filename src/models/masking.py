
import torch

def make_fixed_masks(data, p_mask, seed=999):
    gen = torch.Generator(device=data.y_highway.device)
    gen.manual_seed(seed)

    n = data.num_nodes
    valid_hwy = torch.ones(n, dtype=torch.bool, device=data.y_highway.device)
    valid_lan = (data.y_nlanes != -1)

    # oneway: only valid where we know label (not NaN)
    valid_onw = ~torch.isnan(data.y_oneway)

    valid_wid = ~torch.isnan(data.y_width)
    valid_max = ~torch.isnan(data.y_max)
    valid_min = ~torch.isnan(data.y_min)
    # avg_speed: valid where at least one of the 12 slots is not NaN
    valid_avg = ~torch.isnan(data.y_avg_speed)#.all(dim=1)
    
    # def fixed_mask(valid_mask, p):
    #     r = torch.rand(n, generator=gen, device=valid_mask.device)
    #     return (r < p) & valid_mask
    def fixed_mask(valid_mask, p):
        r = torch.rand(valid_mask.shape, generator=gen, device=valid_mask.device)
        return (r < p) & valid_mask

    return {
        "hwy": fixed_mask(valid_hwy, p_mask),
        "lan": fixed_mask(valid_lan, p_mask),
        "onw": fixed_mask(valid_onw, p_mask),
        "wid": fixed_mask(valid_wid, p_mask),
        "max": fixed_mask(valid_max, p_mask),
        "min": fixed_mask(valid_min, p_mask),
        "avg": fixed_mask(valid_avg, p_mask),
        # NOTE: no "len" mask (length is input-only)
    }
    
def bernoulli_mask(valid_mask, p=0.3):
    r = torch.rand(valid_mask.shape, device=valid_mask.device)
    return (r < p) & valid_mask