from collections import Counter, OrderedDict
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    mean = float(np.mean(x))
    median = float(np.median(x))
    if len(np.unique(x)) == len(x):
        mode =  float(min(x))
    else:
        count = Counter(x)
        mode = float((count.most_common(1)[0][0]))
    return {
        "mean": mean,
        "median": median,
        "mode": mode
    }