def logarithmic_int_range(start, stop, factor):
    steps = []
    while start < stop:
        steps.append(start)
        if int(start * factor) <= start:
            start += 1
        else:
            start *= factor
            start = int(start) + int(start - int(start) > 0)
    return steps
