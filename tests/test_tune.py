from unittest.mock import Mock, patch
import pandas as pd
import optuna
from datetime import datetime
import pytest
from hpobench.config import IntRange, FloatRange, CategoricalRange

from hpobench.tune import (
    set_optuna_params,
    build_optuna_distributions,
    optuna_tune,
    build_confopt_search_space,
    confopt_tune,
    build_skopt_space,
    skopt_tune,
    tune,
)


class TestHelperFunctions:
    def test_set_optuna_params(self):
        trial = Mock()
        trial.suggest_int.return_value = 5
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = "option2"

        params = {
            "int_param": IntRange(type="int", lower=1, upper=10),
            "float_param": FloatRange(type="float", lower=0.1, upper=0.9),
            "cat_param": CategoricalRange(
                type="categorical", choices=["option1", "option2"]
            ),
        }

        result = set_optuna_params(trial, params)

        assert result["int_param"] == 5
        assert result["float_param"] == 0.5
        assert result["cat_param"] == "option2"
        trial.suggest_int.assert_called_once_with("int_param", 1, 10)

    def test_build_optuna_distributions(self):
        params = {
            "int_param": IntRange(type="int", lower=1, upper=10),
            "float_param": FloatRange(type="float", lower=0.1, upper=0.9),
            "cat_param": CategoricalRange(
                type="categorical", choices=["option1", "option2"]
            ),
        }

        dists = build_optuna_distributions(params)

        assert isinstance(
            dists["int_param"], optuna.distributions.IntUniformDistribution
        )
        assert dists["int_param"].low == 1
        assert dists["int_param"].high == 10

    def test_build_confopt_search_space(self):
        params = {
            "int_param": IntRange(type="int", lower=1, upper=3),
            "cat_param": CategoricalRange(
                type="categorical", choices=["option1", "option2"]
            ),
        }

        with patch("random.uniform", return_value=0.5):
            space = build_confopt_search_space(params)
            assert space["int_param"] == [1, 2, 3]
            assert space["cat_param"] == ["option1", "option2"]

    def test_build_skopt_space(self):
        params = {
            "int_param": IntRange(type="int", lower=1, upper=10),
            "cat_param": CategoricalRange(
                type="categorical", choices=["option1", "option2"]
            ),
        }

        space, names = build_skopt_space(params)
        assert len(space) == 2
        assert names == ["int_param", "cat_param"]


class TestOptunaTune:
    @patch("optuna.create_study")
    def test_optuna_tune_basic(self, mock_create_study):
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.datetime_complete = datetime.now()
        mock_trial.value = 0.5
        mock_trial.params = {"param1": 5}
        mock_study.trials = [mock_trial]
        mock_study.best_value = 0.5
        mock_create_study.return_value = mock_study

        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        performance_generator = Mock()
        performance_generator.predict.return_value = 0.5

        history, best_value = optuna_tune(
            params=params,
            performance_generator=performance_generator,
            sampler="tpe",  # Use string literal instead of sampler instance
            n_trials=1,
        )

        assert best_value == 0.5
        assert len(history) == 1
        assert history.iloc[0]["performance"] == 0.5


class TestConfoptTune:
    @patch("hpobench.tune.ObjectiveConformalSearcher")
    def test_confopt_tune_basic(self, mock_searcher_class):
        mock_searcher = Mock()
        mock_trial = Mock(
            timestamp=datetime.now(),
            performance=0.5,
            configuration={"param1": 5},
            breached_interval=False,
            primary_estimator_error=0.1,
            searcher_runtime=0.2,
        )
        mock_searcher.study.trials = [mock_trial]
        mock_searcher_class.return_value = mock_searcher

        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        performance_generator = Mock()
        sampler = Mock()

        with patch("hpobench.tune.build_confopt_search_space") as mock_build:
            mock_build.return_value = {"param1": [1, 2, 3, 4, 5]}
            history, best_value = confopt_tune(
                params=params,
                performance_generator=performance_generator,
                sampler=sampler,
                n_trials=1,
            )

            assert mock_searcher.search.called
            assert len(history) == 1
            assert history.iloc[0]["performance"] == 0.5


class TestSkoptTune:
    @patch("hpobench.tune.gp_minimize")
    def test_skopt_tune_gp(self, mock_gp_minimize):
        mock_result = Mock()
        mock_result.fun = 0.5
        mock_result.func_vals = [0.5]
        mock_result.x_iters = [[5]]
        mock_gp_minimize.return_value = mock_result

        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        performance_generator = Mock()
        performance_generator.predict.return_value = 0.5

        with patch("hpobench.tune.build_skopt_space") as mock_build:
            mock_build.return_value = ([Mock(name="param1")], ["param1"])
            history, best_value = skopt_tune(
                params=params,
                performance_generator=performance_generator,
                sampler="gp",
                n_trials=1,
            )

            assert mock_gp_minimize.called
            assert best_value == 0.5
            assert len(history) == 1

    def test_skopt_tune_unknown_sampler(self):
        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        performance_generator = Mock()

        with pytest.raises(ValueError):
            skopt_tune(
                params=params,
                performance_generator=performance_generator,
                sampler="unknown",
                n_trials=1,
            )


class TestMainTune:
    @patch("hpobench.tune.optuna_tune")
    def test_tune_optuna(self, mock_optuna_tune):
        mock_optuna_tune.return_value = (pd.DataFrame(), 0.5)

        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        tuner_config = Mock()
        tuner_config.tuner = "optuna"
        tuner_config.sampler = "tpe"  # Use string literal instead of sampler instance
        performance_generator = Mock()

        history, best_value = tune(
            performance_generator=performance_generator,
            tuner_config=tuner_config,
            params=params,
            n_trials=1,
        )

        mock_optuna_tune.assert_called_once()
        assert best_value == 0.5

    def test_tune_unknown_tuner(self):
        params = {"param1": IntRange(type="int", lower=1, upper=10)}
        tuner_config = Mock()
        tuner_config.tuner = "unknown"
        performance_generator = Mock()

        with pytest.raises(ValueError):
            tune(
                performance_generator=performance_generator,
                tuner_config=tuner_config,
                params=params,
                n_trials=1,
            )
