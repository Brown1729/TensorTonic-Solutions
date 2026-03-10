def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)
    if len(set_a) == 0 or len(set_b) == 0:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)