def logarithmic_int_range(start, stop, factor, include_stop=False):
    steps = []
    while start < stop:
        steps.append(start)
        if int(start * factor) <= start:
            start += 1
        else:
            start *= factor
            start = int(start) + int(start - int(start) > 0)
    if include_stop:
        steps.append(stop)
    return steps
