def logarithmic_scale_integers(init, factor, max_):
    steps = []
    while init < max_:
        steps.append(init)
        if int(init * factor) <= init:
            init += 1
        else:
            init *= factor
            init = int(init) + int(init - int(init) > 0)
    return steps
