import pandas
import numpy as np

scorer = "dlc_sc"
df = _index = pandas.MultiIndex.from_product(
    [[scorer], ["Glab", "Ment"], ["x", "y", "likelihood"]],
    names=["scorer", "bodyparts", "coords"],
)

_data = np.array([[0, 1, 0.5, 3, 4, 0.99], [6, 7, 0.8, 1, 2, 0.6], [4, 5, 0.9, 7, 8, 0.9]])
df = pandas.DataFrame(_data, columns=_index)
