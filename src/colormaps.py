import matplotlib

cdict = {
    "red": (
        (0.0, 0.0, 0.0),
        (0.25, 0.2, 0.2),
        (0.45, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.55, 0.0, 0.0),
        (0.75, 0.8, 0.8),
        (1.0, 1.0, 1.0),
    ),
    "green": (
        (0.0, 0.0, 1.0),
        (0.25, 0.0, 0.0),
        (0.45, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.55, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ),
    "blue": (
        (0.0, 0.0, 1.0),
        (0.25, 0.8, 0.8),
        (0.45, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.55, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}
ndict = {
    "red": ((0.0, 0.0, 0.0), (0.5, 0.2, 0.2), (0.9, 0.0, 0.0), (1.0, 0.5, 0.5)),
    "green": ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (0.9, 0.0, 0.0), (1.0, 0.5, 0.5)),
    "blue": ((0.0, 0.0, 1.0), (0.5, 0.8, 0.8), (0.9, 0.0, 0.0), (1.0, 0.5, 0.5)),
}
pdict = {
    "red": ((0.0, 0.5, 0.5), (0.1, 0.0, 0.0), (0.5, 0.8, 0.8), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.5, 0.5), (0.1, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 0.5, 0.5), (0.1, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
}

both_cmap = matplotlib.colors.LinearSegmentedColormap("brain_combined", cdict, 256)
pos_cmap = matplotlib.colors.LinearSegmentedColormap("brain_above", pdict, 256)
neg_cmap = matplotlib.colors.LinearSegmentedColormap("brain_below", ndict, 256)

CMAPS = {
    "both": both_cmap,
    "neg": neg_cmap,
    "pos": pos_cmap,
}

# dict of signs for filtering negative/positive components ("both" is here for compatibility)
# it is intentionally inverted
DSGN = {
    "both": 0.0,
    "neg": 1.0,
    "pos": -1.0,
}