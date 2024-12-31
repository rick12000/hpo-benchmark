import random
from typing import Tuple, Optional, Any
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler

def train_val_split(
    X: np.array,
    y: np.array,
    train_split: float,
    normalize: bool = True,
    ordinal: bool = False,
    random_state: int = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Split X and y data into training and validation sets.

    Splits can be carried out randomly or sequentially, with or without normalization.

    Parameters
    ----------
    X :
        Feature variables.
    y :
        Target variable.
    train_split :
        Percentage of training data to carve out of the overall
        data. Values must be contained in the [0, 1] interval.
    normalize :
        Whether X features in both the training and validation
        splits should be normalized according to the training split.
    ordinal :
        Whether the split should occur ordinally (only set to True
        if the X and y data was passed according to some sequential
        order, eg. sorted by date), else split will be random.
    random_state :
        Random seed.

    Returns
    -------
    X_train :
        X features training split.
    y_train :
        y target training split.
    X_val :
        X features validation split.
    y_val :
        y target validation split.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if ordinal:
        train_idx = list(range(len(X) - round(len(X) * (1 - train_split))))
        val_idx = list(range(len(X) - round(len(X) * (1 - train_split)), len(X)))
    else:
        train_idx = list(
            np.random.choice(len(X), round(len(X) * train_split), replace=False)
        )
        val_idx = list(np.setdiff1d(np.arange(len(X)), train_idx))

    X_val = X[val_idx, :]
    X_train = X[train_idx, :]

    y_val = y[val_idx]
    y_train = y[train_idx]

    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val

def update_model_parameters(
    model_instance, configuration, random_state
):
    """
    Updates the attributes of an initialized model object.

    Only attributes which are specified in the 'configuration'
    dictionary input of this function will be overridden.

    Parameters
    ----------
    model_instance :
        An instance of a prediction model.
    configuration :
        A dictionary whose keys represent the attributes of
        the model instance that need to be overridden and whose
        values represent what they should be overridden to.
        Keys must match model instance attribute names.
    random_state :
        Random generation seed.

    Returns
    -------
    updated_model_instance :
        Model instance with updated attributes.
    """
    updated_model_instance = deepcopy(model_instance)
    for tuning_attr_name, tuning_attr in configuration.items():
        setattr(updated_model_instance, tuning_attr_name, tuning_attr)
    if hasattr(updated_model_instance, "random_state"):
        setattr(updated_model_instance, "random_state", random_state)
    return updated_model_instance

