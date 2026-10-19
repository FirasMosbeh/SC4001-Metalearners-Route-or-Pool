def coarse_to_fine(n):
    """Order the indices 0..n-1 so that the first k of them are spread evenly over the range, for any k.

    Each round splits the range into twice as many equal parts and adds the start of every
    part that isn't in the order yet:

        1 part:   0
        2 parts:  n/2
        4 parts:  n/4, 3n/4
        8 parts:  n/8, 3n/8, 5n/8, 7n/8
        ...       until every index is in the order

    Example, n = 16: 0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15

    Used for shards: rows are sorted by label and class, and we stop downloading as soon
    as all pools are full. Wherever we stop, the shards visited so far cover the whole
    range of labels and classes instead of just the beginning.
    """
    order, already_added, parts = [], set(), 1
    while len(order) < n:
        for part in range(parts):
            index = part * n // parts
            if index not in already_added:
                already_added.add(index)
                order.append(index)
        parts *= 2
    return order
