from sklearn.ensemble import RandomForestRegressor  # GradientBoostingRegressor
import pandas as pd
from tune import tune
from datetime import datetime
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
import random

# import json
import logging
import optuna
from generate import generate_data

from copy import deepcopy

from scipy.stats import skew, kurtosis
from scipy.stats import spearmanr


from sklearn.model_selection import GroupShuffleSplit

from typing import Dict, List, Any
from itertools import combinations

from sklearn.feature_selection import mutual_info_regression

from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_sparsity(X):
    """
    Calculates the sparsity of a NumPy array.

    Args:
      X: The input NumPy array.

    Returns:
      The sparsity of the array, defined as the proportion of zero elements.
    """
    num_zeros = np.count_nonzero(X == 0)
    total_elements = X.size
    return num_zeros / total_elements


def calculate_vif(X):
    return np.mean(
        np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    )


def total_mutual_information(X):
    # Calculate mutual information between all feature pairs
    mi_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            mi_matrix[i, j] = mutual_info_regression(X[:, [i]], X[:, j])[0]
    return np.mean(np.abs(mi_matrix))


def generate_param_combinations(
    param_sample_space: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    # Get all parameter keys
    keys = list(param_sample_space.keys())

    # Generate all possible combination sizes (1 to full dictionary)
    all_combinations = []

    for r in range(1, len(keys) + 1):
        # Generate combinations of keys of length r
        for key_combo in combinations(keys, r):
            # Create a dictionary for this combination
            combo_dict = {}

            # Add the parameters for this combination
            for key in key_combo:
                combo_dict[key] = param_sample_space[key]

            all_combinations.append(combo_dict)

    return all_combinations


# Find the most common rank order across all rank groups
def get_most_common_rank_order(df):
    # Group by rank_group and get the order of tuners for each group
    rank_orders = df.groupby("rank_group").apply(
        lambda x: tuple(x.sort_values("prediction_rank")["tuner"].tolist())
    )

    # Find the most common rank order
    most_common_order = rank_orders.mode().iloc[0]
    return most_common_order


# Create a naive ranker that always predicts the most common rank order
def naive_ranker_predict(df, most_common_order):
    naive_predictions = []

    for group in df["rank_group"].unique():
        group_df = df[df["rank_group"] == group].copy()

        # Create a DataFrame with the most common order
        naive_rank_df = pd.DataFrame(
            {
                "tuner": most_common_order,
                "predicted_rank": range(1, len(most_common_order) + 1),
            }
        )

        # Merge with the original group DataFrame
        group_df = group_df.merge(naive_rank_df, on="tuner", how="left")
        naive_predictions.append(group_df)

    naive_prediction_set = pd.concat(naive_predictions)

    return naive_prediction_set


def calculate_average_precision_at_k(df, k=1):
    """Calculates the average Precision@k across rank groups.

    Args:
        df: Pandas DataFrame with 'rank_group', 'rank', and 'predicted_rank' columns.
        k: The value of k for Precision@k.

    Returns:
        The average Precision@k score across all rank groups, or NaN if no groups are found.
        Prints out the precision at k for each group.
    """
    precision_scores = []
    for group in df["rank_group"].unique():
        group_df = df[df["rank_group"] == group].copy()

        # Sort by predicted rank
        group_df.sort_values(by="predicted_rank", inplace=True)

        # Get the top k predicted ranks
        top_k_predicted = group_df.iloc[:k]

        # Calculate how many of the top k predicted ranks are in the top k actual ranks
        group_df.sort_values(by="prediction_rank", inplace=True)
        top_k_actual = group_df.iloc[:k]

        relevant_items = set(top_k_actual["tuner"]).intersection(
            set(top_k_predicted["tuner"])
        )

        precision = len(relevant_items) / k if k <= len(top_k_predicted) else 0.0
        precision_scores.append(precision)

    if not precision_scores:
        return np.nan
    return np.mean(precision_scores)


class RankerXGBoost:
    def __init__(
        self, learning_rate=0.01, max_depth=3, num_boost_round=250, random_state=42
    ):
        self.params = {
            "objective": "rank:pairwise",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "seed": random_state,
        }
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, X, y, groups):
        train_dmatrix = xgb.DMatrix(X, label=y)
        train_dmatrix.set_group(groups)

        self.model = xgb.train(
            self.params, train_dmatrix, num_boost_round=self.num_boost_round
        )

    def predict(self, X):
        if not self.model:
            raise ValueError("Model has not been trained. Call fit() first.")

        test_dmatrix = xgb.DMatrix(X)
        y_pred = self.model.predict(test_dmatrix)
        return y_pred

    def get_feature_importances(self, feature_names=None, plot=True):
        if not self.model:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Extract feature importances
        importances = self.model.get_score(importance_type="weight")

        # Convert to DataFrame if feature names are provided
        if feature_names is not None:
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": [
                        importances.get(f"f{i}", 0) for i in range(len(feature_names))
                    ],
                }
            ).sort_values("importance", ascending=False)
        else:
            importance_df = pd.DataFrame.from_dict(
                importances, orient="index", columns=["importance"]
            ).reset_index()
            importance_df.columns = ["feature", "importance"]
            importance_df = importance_df.sort_values("importance", ascending=False)

        # Plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.bar(importance_df["feature"], importance_df["importance"])
            plt.title("XGBoost Feature Importances")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        return importance_df


cache_path = "cache/"
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

run_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_path = cache_path + f"logs/{run_start}"
if not os.path.exists(log_path):
    os.makedirs(log_path)
LOG_FILENAME = (
    f"{log_path}/run_{datetime.now().strftime(format='%m_%d_%Y-%H_%M_%S')}.log"
)
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("hyperopt").setLevel(logging.ERROR)
logging.getLogger("confopt").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

train_split = 0.75
normalize = True
random_state = 1234
n_repetitions = 1

default_toy_data_params = {
    "n_samples": 1000,
    "n_x_features": 15,
    "n_y_features": 1,
    "noise_level": 0.25,
    "n_redundant_linear": 0,
    "n_redundant_noise": 0,
    "sparsity": 1,
    "transformer": "linear",
    "random_state": 1234,
    "to_array": True,
}

dataset_configs = []
noise_level_values = [0, 0.1, 1, 5]
sparsity_level_values = [0.1, 1]
n_x_features_values = [5, 30]
n_samples_values = [500, 100]
for noise_level in noise_level_values:
    for sparsity_level in sparsity_level_values:
        for n_x_features in n_x_features_values:
            for n_samples in n_samples_values:
                toy_data_params = deepcopy(default_toy_data_params)
                toy_data_params["noise_level"] = noise_level
                toy_data_params["sparsity"] = sparsity_level
                toy_data_params["n_x_features"] = n_x_features
                toy_data_params["n_samples"] = n_samples

                # Call the function using the dictionary unpacking syntax
                toy_data = generate_data(**toy_data_params)
                mutual_information = total_mutual_information(X=toy_data[0])
                vif = calculate_vif(X=toy_data[0])
                y_skew = skew(toy_data[1])
                y_kurtosis = kurtosis(toy_data[1])
                y_rel_std = np.std(toy_data[1]) / np.mean(toy_data[1])
                corrs = []
                for i in range(toy_data[0].shape[1]):
                    corr = spearmanr(toy_data[0][:, i], toy_data[1])[0]
                    corrs.append(abs(corr))
                avg_corr = np.mean(np.array(corrs))
                max_corr = max(corrs)

                dataset_configs.append(
                    {
                        "name": f"TOY_NOISE_{str(noise_level).replace('.', 'dec')}",
                        "data": toy_data,
                        "synthetic_params": toy_data_params,
                        "data_attributes": {
                            "mutual_information": mutual_information,
                            "vif": vif,
                            "y_skew": y_skew,
                            "y_kurtosis": y_kurtosis,
                            "y_rel_std": y_rel_std,
                            "avg_corr": avg_corr,
                            "max_corr": max_corr,
                        },
                        "normalize": True,
                        "evaluation_metric": "mean_squared_error",
                        "evaluation_metric_direction": "inverse",
                        "timeout": 60 * 2,
                    }
                )
param_sample_space = {
    "n_estimators__range_int": [10, 400],
    "min_samples_split__range_float": [0.005, 0.3],
    "min_samples_leaf__range_float": [0.005, 0.3],
    # "max_features__range_float": [0.1, 1],
    # "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}
params_combinations = generate_param_combinations(param_sample_space=param_sample_space)

model_configs = []

for params in params_combinations:
    n_params = len(params)
    n_int_params = sum([1 if "int" in x else 0 for x in list(params.keys())])
    n_float_params = sum([1 if "float" in x else 0 for x in list(params.keys())])
    n_categorical_params = sum(
        [1 if ("int" not in x and "float" not in x) else 0 for x in list(params.keys())]
    )
    model_configs.append(
        {
            "model_name": "Random Forest",
            "model": RandomForestRegressor(),
            "params": params,
            "param_attributes": {
                "n_params": n_params,
                "n_int_params": n_int_params,
                "n_float_params": n_float_params,
                "n_categorical_params": n_categorical_params,
            },
        }
    )

model_configs = random.sample(model_configs, k=10)
tuners = ["confopt-qgbm-0.9", "optuna-tpe", "hyperopt-random", "optuna-cmaes"]

# tuners = ["confopt", "optuna-tpe", "optuna-cmaes", "hyperopt-tpe", "hyperopt-random"]

feature_names = [
    "sparsity",
    "n_x_features",
    "n_samples",
    "mutual_information",
    "vif",
    "y_skew",
    "y_kurtosis",
    "y_rel_std",
    "avg_corr",
    "max_corr",
    "n_params",
    "n_int_params",
    "n_float_params",
    "n_categorical_params",
]

raw_benchmark_data = pd.DataFrame()

logger.info("Running HPO benchmark...")
for dataset_config in dataset_configs:
    dataset_name = dataset_config["name"]
    logger.info(f"Dataset: {dataset_name}")
    metric_direction = dataset_config["evaluation_metric_direction"]
    timeout = dataset_config["timeout"]
    X, y = dataset_config["data"]
    for config in model_configs:
        logger.info(f"Model: {config['model_name']}")
        for tuner in tuners:
            logger.info(f"Tuner: {tuner}")
            for repetition in range(n_repetitions):
                logger.info(f"Repetition: {repetition}")
                tune_start = datetime.now()
                historical_performance, best_value = tune(
                    model=config["model"],
                    X=X,
                    y=y,
                    train_split=train_split,
                    normalize=normalize,
                    tuner=tuner,
                    timeout=timeout,
                    random_state=random_state,
                    params=config["params"],
                )
                historical_performance["runtime"] = (
                    historical_performance["end_time"] - tune_start
                ).dt.seconds

                historical_performance["dataset"] = dataset_name
                historical_performance["sparsity"] = dataset_config["synthetic_params"][
                    "sparsity"
                ]
                historical_performance["noise_level"] = dataset_config[
                    "synthetic_params"
                ]["noise_level"]
                historical_performance["n_x_features"] = dataset_config[
                    "synthetic_params"
                ]["n_x_features"]
                historical_performance["n_samples"] = dataset_config[
                    "synthetic_params"
                ]["n_samples"]
                historical_performance["mutual_information"] = dataset_config[
                    "data_attributes"
                ]["mutual_information"]
                historical_performance["vif"] = dataset_config["data_attributes"]["vif"]
                historical_performance["y_skew"] = dataset_config["data_attributes"][
                    "y_skew"
                ]
                historical_performance["y_kurtosis"] = dataset_config[
                    "data_attributes"
                ]["y_kurtosis"]
                historical_performance["y_rel_std"] = dataset_config["data_attributes"][
                    "y_rel_std"
                ]
                historical_performance["avg_corr"] = dataset_config["data_attributes"][
                    "avg_corr"
                ]
                historical_performance["max_corr"] = dataset_config["data_attributes"][
                    "max_corr"
                ]
                historical_performance["n_params"] = config["param_attributes"][
                    "n_params"
                ]
                historical_performance["n_int_params"] = config["param_attributes"][
                    "n_int_params"
                ]
                historical_performance["n_float_params"] = config["param_attributes"][
                    "n_float_params"
                ]
                historical_performance["n_categorical_params"] = config[
                    "param_attributes"
                ]["n_categorical_params"]
                historical_performance["model"] = config["model_name"]
                historical_performance["tuner"] = tuner
                historical_performance["repetition"] = repetition + 1

                if metric_direction == "direct":
                    time_grouping_aggregation = "max"
                elif metric_direction == "inverse":
                    time_grouping_aggregation = "min"
                historical_performance_second_aggregated = (
                    historical_performance.groupby(
                        ["dataset", "model", "repetition", "runtime", "tuner"]
                        + feature_names,
                        as_index=False,
                    ).agg({"performance": time_grouping_aggregation})
                )

                historical_performance_second_aggregated = (
                    historical_performance_second_aggregated.sort_values(
                        by=[
                            "dataset",
                            "model",
                            "tuner",
                            "repetition",
                            "runtime",
                        ]
                        + feature_names,
                        ascending=True,
                    )
                )
                historical_performance_second_aggregated[
                    "best_performance"
                ] = historical_performance_second_aggregated.groupby(
                    [
                        "dataset",
                        "model",
                        "tuner",
                        "repetition",
                    ]
                    + feature_names
                )[
                    "performance"
                ].transform(
                    "cum" + time_grouping_aggregation
                )

                # Expand runtime grid of search report:
                runtime_spacings = pd.DataFrame(
                    {
                        "runtime": np.arange(
                            1,
                            timeout,
                            1,
                        )
                    }
                ).astype(int)
                historical_performance_second_filled = pd.merge(
                    runtime_spacings,
                    historical_performance_second_aggregated,
                    how="left",
                    on="runtime",
                ).ffill()  # NOTE: ffill to propagate values until a new runtime filled value is available

                raw_benchmark_data = pd.concat(
                    [raw_benchmark_data, historical_performance_second_filled], axis=0
                )

                data_path = cache_path + f"data/{run_start}"
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                raw_benchmark_data.to_csv(
                    f"{data_path}/incremental_raw_benchmark_data.csv", index=False
                )

# Apply ranks within each repetition and tuner:
ascending = True  # False for accuracy
raw_benchmark_data["rank"] = raw_benchmark_data.groupby(
    [
        "dataset",
        "model",
        "repetition",
        "runtime",
    ]
    + feature_names,
    as_index=False,
)["best_performance"].rank(method="average", ascending=ascending)

# Calculate average time weighted rank across repetitions and time steps:
processed_benchmark_data = raw_benchmark_data.groupby(
    [
        "dataset",
        "model",
        "tuner",
    ]
    + feature_names,
    as_index=False,
).agg({"rank": "mean"})

# data_path = cache_path + f"data/{run_start}"
# if not os.path.exists(data_path):
#     os.makedirs(data_path)
# processed_benchmark_data.to_csv(
#     f"{data_path}/data_attribute_benchmark_data.csv", index=False
# )
# raw_benchmark_data.to_csv(f"{data_path}/data_attribute_raw_data.csv", index=False)


# Assumes only one model is benchmarked at a time:
processed_benchmark_data["rank_group"] = (
    processed_benchmark_data[feature_names + ["dataset"]]
    .astype(str)
    .agg("-".join, axis=1)
)

processed_benchmark_data["prediction_rank"] = processed_benchmark_data.groupby(
    [
        "dataset",
        "model",
    ]
    + feature_names,
    as_index=False,
)["rank"].rank(method="average", ascending=ascending)

# Initialize cross-validation parameters
n_splits = 10
splitter = GroupShuffleSplit(test_size=0.1, n_splits=n_splits, random_state=None)

# Arrays to store precision at 1 for each split
naive_precisions = []
xgboost_precisions = []

for train_inds, test_inds in splitter.split(
    processed_benchmark_data, groups=processed_benchmark_data["rank_group"]
):

    benchmark_data_train = processed_benchmark_data.iloc[train_inds]
    benchmark_data_train = benchmark_data_train.sort_values(
        by=["rank_group"]
    ).reset_index()
    benchmark_data_train_X = benchmark_data_train[feature_names + ["tuner"]]
    benchmark_data_train_X = pd.concat(
        [
            benchmark_data_train_X,
            pd.get_dummies(benchmark_data_train["tuner"]).astype(int),
        ],
        axis=1,
    ).drop(["tuner"], axis=1)
    benchmark_data_train_y = benchmark_data_train["prediction_rank"]

    benchmark_data_val = processed_benchmark_data.iloc[test_inds]
    benchmark_data_val = benchmark_data_val.sort_values(by=["rank_group"]).reset_index()
    benchmark_data_val_X = benchmark_data_val[feature_names + ["tuner"]]
    benchmark_data_val_X = pd.concat(
        [
            benchmark_data_val_X,
            pd.get_dummies(benchmark_data_val_X["tuner"]).astype(int),
        ],
        axis=1,
    ).drop(["tuner"], axis=1)
    benchmark_data_val_y = benchmark_data_val["prediction_rank"]

    ranker = RankerXGBoost(
        learning_rate=0.01, max_depth=None, num_boost_round=1000, random_state=1234
    )
    groups = [len(tuners)] * len(benchmark_data_train["rank_group"].unique())
    ranker.fit(
        benchmark_data_train_X.to_numpy(), benchmark_data_train_y.to_numpy(), groups
    )
    y_pred = ranker.predict(benchmark_data_val_X.to_numpy())

    prediction_set = benchmark_data_val[["rank_group", "tuner", "prediction_rank"]]
    prediction_set["predicted_rank"] = y_pred

    # Get the most common rank order
    most_common_order = get_most_common_rank_order(benchmark_data_train)

    # Create the naive prediction set
    naive_prediction_set = naive_ranker_predict(
        benchmark_data_val[["rank_group", "tuner", "prediction_rank"]],
        most_common_order,
    )

    # Calculate Precision@k for the naive ranker
    naive_precision_at_1 = calculate_average_precision_at_k(
        df=naive_prediction_set, k=1
    )

    # Compare with the XGBoost ranker
    xgboost_precision_at_1 = calculate_average_precision_at_k(df=prediction_set, k=1)

    naive_precisions.append(naive_precision_at_1)
    xgboost_precisions.append(xgboost_precision_at_1)

naive_predictions_mean = np.mean(np.array(naive_precisions))
naive_predictions_std = np.std(np.array(naive_precisions))
print(f"Naive predictions mean: {naive_predictions_mean}, std: {naive_predictions_std}")
xgboost_precisions_mean = np.mean(np.array(xgboost_precisions))
xgboost_precisions_std = np.std(np.array(xgboost_precisions))
print(
    f"XGBoost predictions mean: {xgboost_precisions_mean}, std: {xgboost_precisions_std}"
)


ranker.get_feature_importances(feature_names=feature_names + ["tuner"])

#########

cali_data = fetch_california_housing(return_X_y=True)
diabetes_data = load_diabetes(return_X_y=True)
public_dataset_configs = [
    {
        "name": "CALI",
        "data": cali_data,
        "normalize": True,
        "evaluation_metric": "mean_squared_error",
        "evaluation_metric_direction": "inverse",
        "timeout": 60 * 2,
    },
    {
        "name": "DIABETES",
        "data": diabetes_data,
        "normalize": True,
        "evaluation_metric": "mean_squared_error",
        "evaluation_metric_direction": "inverse",
        "timeout": 60,
    },
]

public_prediction_set = []
for dataset_config in public_dataset_configs:
    data = dataset_config["data"]
    mutual_information = total_mutual_information(X=data[0])
    vif = calculate_vif(X=data[0])
    y_skew = skew(data[1])
    y_kurtosis = kurtosis(data[1])
    y_rel_std = np.std(data[1]) / np.mean(data[1])
    corrs = []
    for i in range(data[0].shape[1]):
        corr = spearmanr(data[0][:, i], data[1])[0]
        corrs.append(abs(corr))
    avg_corr = np.mean(np.array(corrs))
    max_corr = max(corrs)
    sparsity = get_sparsity(data[0])

    # TODO: hard coded:
    params = model_configs[2]
    n_params = len(params) + 1
    n_int_params = sum([1 if "int" in x else 0 for x in list(params.keys())])
    n_float_params = sum([1 if "float" in x else 0 for x in list(params.keys())])
    n_categorical_params = sum(
        [1 if ("int" not in x and "float" not in x) else 0 for x in list(params.keys())]
    )

    for tuner in tuners:
        public_prediction_set.append(
            {
                "sparsity": sparsity,
                "n_x_features": data[0].shape[1],
                "n_samples": len(data[0]),
                "mutual_information": mutual_information,
                "vif": vif,
                "y_skew": y_skew,
                "y_kurtosis": y_kurtosis,
                "y_rel_std": y_rel_std,
                "avg_corr": avg_corr,
                "max_corr": max_corr,
                "n_params": n_params,
                "n_int_params": n_int_params,
                "n_float_params": n_float_params,
                "n_categorical_params": n_categorical_params,
                "tuner": tuner,
            }
        )

df_public_prediction_set = pd.DataFrame(public_prediction_set)[
    feature_names + ["tuner"]
]
df_public_prediction_set = pd.concat(
    [
        df_public_prediction_set,
        pd.get_dummies(df_public_prediction_set["tuner"]).astype(int),
    ],
    axis=1,
).drop(["tuner"], axis=1)


public_y_pred = ranker.predict(df_public_prediction_set.to_numpy())
print(public_y_pred)
