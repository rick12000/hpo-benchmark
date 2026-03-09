from hpobench.generation.generate import BlackBoxGenerator


def test_blackbox_generator_predict_reproducibility():
    generator = BlackBoxGenerator(generator="rastrigin")
    configuration = {"param1": 1.0, "param2": 2.0, "param3": 3.0}

    result1 = generator.predict(configuration)
    result2 = generator.predict(configuration)

    assert result1 == result2

    result1 = generator.predict_runtime(configuration)
    result2 = generator.predict_runtime(configuration)

    assert result1 == result2
