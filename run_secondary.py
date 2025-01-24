from sklearn.ensemble import RandomForestRegressor  # GradientBoostingRegressor
import pandas as pd
from tune import tune
from datetime import datetime
import os
import xgboost as xgb
import numpy as np

# import json
import logging
import optuna
from generate import generate_data

from copy import deepcopy


from sklearn.model_selection import GroupShuffleSplit


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
        group_df.sort_values(by="rank", inplace=True)
        top_k_actual = group_df.iloc[:k]

        relevant_items = set(top_k_actual["tuner"]).intersection(
            set(top_k_predicted["tuner"])
        )

        precision = len(relevant_items) / k if k <= len(top_k_predicted) else 0.0
        print(f"Precision@{k} for group {group}: {precision}")
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
    "n_redundant_noise": 5,
    "sparsity": 1,
    "transformer": "linear",
    "random_state": 1234,
    "to_array": True,
}

dataset_configs = []
noise_level_values = [0, 0.2, 0.5, 1, 5]
sparsity_level_values = [0.01, 0.1, 0.5, 0.75, 1]
for noise_level in noise_level_values:
    for sparsity_level in sparsity_level_values:
        toy_data_params = deepcopy(default_toy_data_params)
        toy_data_params["noise_level"] = noise_level
        toy_data_params["sparsity"] = sparsity_level

        # Call the function using the dictionary unpacking syntax
        toy_data = generate_data(**toy_data_params)

        dataset_configs.append(
            {
                "name": f"TOY_NOISE_{str(noise_level).replace('.', 'dec')}",
                "data": toy_data,
                "synthetic_params": toy_data_params,
                "normalize": True,
                "evaluation_metric": "mean_squared_error",
                "evaluation_metric_direction": "inverse",
                "timeout": 30,
            }
        )

model_configs = [
    {
        "model_name": "Random Forest",
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators__range_int": [10, 400],
            "min_samples_split__range_float": [0.005, 0.3],
            "min_samples_leaf__range_float": [0.005, 0.3],
            "max_features__range_float": [0.1, 1],
        },
    },
    # {
    #     "model_name": "Gradient Boosting Machine",
    #     "model": GradientBoostingRegressor(),
    #     "params": {
    #         "learning_rate": [0.001, 0.01, 0.1],
    #         "n_estimators": [10, 30, 50, 100, 150, 200, 300, 400],
    #         "min_samples_split": [0.005, 0.01, 0.1, 0.2, 0.3],
    #         "min_samples_leaf": [0.005, 0.01, 0.1, 0.2, 0.3],
    #         "max_features": [None, 0.8, 0.9, 1],
    #     },
    # },
]
tuners = ["optuna-cmaes", "confopt-qgbm-0.9", "optuna-tpe", "hyperopt-random"]

# tuners = ["confopt", "optuna-tpe", "optuna-cmaes", "hyperopt-tpe", "hyperopt-random"]

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
                historical_performance["model"] = config["model_name"]
                historical_performance["tuner"] = tuner
                historical_performance["repetition"] = repetition + 1

                if metric_direction == "direct":
                    time_grouping_aggregation = "max"
                elif metric_direction == "inverse":
                    time_grouping_aggregation = "min"
                historical_performance_second_aggregated = (
                    historical_performance.groupby(
                        [
                            "dataset",
                            "sparsity",
                            "noise_level",
                            "model",
                            "tuner",
                            "repetition",
                            "runtime",
                        ],
                        as_index=False,
                    ).agg({"performance": time_grouping_aggregation})
                )

                historical_performance_second_aggregated = (
                    historical_performance_second_aggregated.sort_values(
                        by=[
                            "dataset",
                            "sparsity",
                            "noise_level",
                            "model",
                            "tuner",
                            "repetition",
                            "runtime",
                        ],
                        ascending=True,
                    )
                )
                historical_performance_second_aggregated[
                    "best_performance"
                ] = historical_performance_second_aggregated.groupby(
                    [
                        "dataset",
                        "sparsity",
                        "noise_level",
                        "model",
                        "tuner",
                        "repetition",
                    ]
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
    ["dataset", "sparsity", "noise_level", "model", "repetition", "runtime"],
    as_index=False,
)["best_performance"].rank(method="average", ascending=ascending)

# Calculate average time weighted rank across repetitions and time steps:
processed_benchmark_data = raw_benchmark_data.groupby(
    ["dataset", "sparsity", "noise_level", "model", "tuner"], as_index=False
).agg({"rank": "mean"})

data_path = cache_path + f"data/{run_start}"
if not os.path.exists(data_path):
    os.makedirs(data_path)
processed_benchmark_data.to_csv(
    f"{data_path}/data_attribute_benchmark_data.csv", index=False
)
raw_benchmark_data.to_csv(f"{data_path}/data_attribute_raw_data.csv", index=False)


# Assumes only one model is benchmarked at a time:
processed_benchmark_data["rank_group"] = (
    processed_benchmark_data["dataset"]
    + processed_benchmark_data["sparsity"].astype(str)
    + processed_benchmark_data["noise_level"].astype(str)
)
splitter = GroupShuffleSplit(test_size=0.1, n_splits=2, random_state=None)
split = splitter.split(
    processed_benchmark_data, groups=processed_benchmark_data["rank_group"]
)
train_inds, test_inds = next(split)

benchmark_data_train = processed_benchmark_data.iloc[train_inds]
benchmark_data_train = benchmark_data_train.sort_values(by=["rank_group"]).reset_index()
benchmark_data_train_X = benchmark_data_train[["sparsity", "noise_level", "tuner"]]
benchmark_data_train_X = pd.concat(
    [benchmark_data_train_X, pd.get_dummies(benchmark_data_train["tuner"]).astype(int)],
    axis=1,
).drop(["tuner"], axis=1)
benchmark_data_train_y = benchmark_data_train[["rank"]]

benchmark_data_val = processed_benchmark_data.iloc[test_inds]
benchmark_data_val = benchmark_data_val.sort_values(by=["rank_group"]).reset_index()
benchmark_data_val_X = benchmark_data_val[["sparsity", "noise_level", "tuner"]]
benchmark_data_val_X = pd.concat(
    [benchmark_data_val_X, pd.get_dummies(benchmark_data_val_X["tuner"]).astype(int)],
    axis=1,
).drop(["tuner"], axis=1)
benchmark_data_val_y = benchmark_data_val[["rank"]]


ranker = RankerXGBoost()
groups = [len(tuners)] * len(benchmark_data_train["rank_group"].unique())
ranker.fit(benchmark_data_train_X.to_numpy(), benchmark_data_train_y.to_numpy(), groups)
y_pred = ranker.predict(benchmark_data_val_X.to_numpy())

prediction_set = benchmark_data_val[["rank_group", "tuner", "rank"]]
prediction_set["predicted_rank"] = y_pred


calculate_average_precision_at_k(df=prediction_set, k=1)
