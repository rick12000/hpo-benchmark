# avg_runtimes = []
# for config, id in zip(lc_bench_configs,OPEN_ML_IDS):
#     generator = config.generator
#     combinations = generate_hyperparameter_combinations(
#         params=config.search_space, n_combinations=50, random_state=1234)
#     runtimes = []
#     for combination in combinations:
#         runtime = generator.predict_runtime(configuration=combination)
#         runtimes.append(runtime)
#     avg_runtime = np.mean(np.array(runtimes))
#     avg_runtimes.append(avg_runtime)

# id_runtimes = [{"id": id, "avg_runtime": avg_runtime} for id, avg_runtime in zip(OPEN_ML_IDS, avg_runtimes)]
# df_id_runtimes = pd.DataFrame(id_runtimes)
# print(df_id_runtimes)
# df_id_runtimes[df_id_runtimes["avg_runtime"] <= 120]["id"].tolist()
