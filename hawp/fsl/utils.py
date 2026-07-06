def reached_debug_limit(max_iters, step):
    return max_iters is not None and step >= max_iters
