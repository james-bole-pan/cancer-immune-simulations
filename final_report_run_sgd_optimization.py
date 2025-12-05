import autograd.numpy as anp

def weighted_bce_with_logits(z, y, w_pos=2.333, w_neg=1.0):
    # default weights correspond to NR:R ratio of 7:3
    return (w_pos * y) * anp.log1p(anp.exp(-z)) + (w_neg * (1 - y)) * anp.log1p(anp.exp(z))

