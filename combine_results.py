import pandas as pd

if __name__ == '__main__':

    # building table for result 1

    shs27_bfs = pd.read_csv('save_test_results/SHS27k_bfs', sep=',', header=None)
    shs27_dfs = pd.read_csv('save_test_results/SHS27k_dfs', sep=',', header=None)
    shs27_random = pd.read_csv('save_test_results/SHS27k_random', sep=',', header=None)

    shs148_bfs = pd.read_csv('save_test_results/SHS148k_bfs', sep=',', header=None)
    shs148_dfs = pd.read_csv('save_test_results/SHS148k_dfs', sep=',', header=None)
    shs148_random = pd.read_csv('save_test_results/SHS148k_random', sep=',', header=None)

    cols = ['X_all', 'X_bs', 'X_es', 'X_ns', 'bs', 'es', 'ns']
    means = pd.DataFrame(columns=cols)
    stds = pd.DataFrame(columns=cols)
    for table in [shs27_random, shs27_bfs, shs27_dfs, shs148_random, shs148_bfs, shs148_dfs]:
        table[1] = table[1].replace(' None', -1)
        means = means.append({k: v for k, v in zip(cols, table.mean())}, ignore_index=True)
        stds = stds.append({k: v for k, v in zip(cols, table.std())}, ignore_index=True)


    means.index = ['random', 'bfs', 'dfs','random', 'bfs', 'dfs']

    stds.index = ['random', 'bfs', 'dfs','random', 'bfs', 'dfs']

    combined = means.round(2).astype(str) + ' pm ' + stds.round(2).astype(str)
    print(combined.to_latex())

    # result 2
    gen_shs27_bfs = pd.read_csv('save_test_results/generalize_SHS27k_bfs', sep=',', header=None)
    gen_shs27_dfs = pd.read_csv('save_test_results/generalize_SHS27k_dfs', sep=',', header=None)
    gen_shs27_random = pd.read_csv('save_test_results/generalize_SHS27k_random', sep=',', header=None)

    gen_shs148_bfs = pd.read_csv('save_test_results/generalize_SHS148k_bfs', sep=',', header=None)
    gen_shs148_dfs = pd.read_csv('save_test_results/generalize_SHS148k_dfs', sep=',', header=None)
    gen_shs148_random = pd.read_csv('save_test_results/generalize_SHS148k_random', sep=',', header=None)

    cols = ['random', 'BFS', 'DFS']
    gen_means = pd.DataFrame(columns=cols)
    gen_stds = pd.DataFrame(columns=cols)

    gen_means = gen_means.append({'random': gen_shs27_random.mean().values[0], 'BFS': gen_shs27_bfs.mean().values[0],
                                  'DFS': gen_shs27_dfs.mean().values[0]}, ignore_index=True)
    gen_means = gen_means.append({'random': gen_shs148_random.mean().values[0], 'BFS': gen_shs148_bfs.mean().values[0],
                                  'DFS': gen_shs148_dfs.mean().values[0]}, ignore_index=True)
    gen_stds = gen_stds.append({'random': gen_shs27_random.std().values[0], 'BFS': gen_shs27_bfs.std().values[0],
                                'DFS': gen_shs27_dfs.std().values[0]}, ignore_index=True)
    gen_stds = gen_stds.append({'random': gen_shs148_random.std().values[0], 'BFS': gen_shs148_bfs.std().values[0],
                                'DFS': gen_shs148_dfs.std().values[0]}, ignore_index=True)

    gen_means.index = ['SHS27k-Train', 'SHS148k-Train']

    gen_stds.index = ['SHS27k-Train', 'SHS148k-Train']

    gen_combined = gen_means.round(2).astype(str) + ' pm ' + gen_stds.round(2).astype(str)
    print(gen_combined.to_latex())

    # result 3
    gct_bfs_27 = pd.read_csv('save_test_results/all_GCT_SHS27k_bfs', sep=',', header=None)
    gct_dfs_27 = pd.read_csv('save_test_results/all_GCT_SHS27k_dfs', sep=',', header=None)
    gct_bfs_148 = pd.read_csv('save_test_results/all_GCT_SHS148k_bfs', sep=',', header=None)
    gct_dfs_148 = pd.read_csv('save_test_results/all_GCT_SHS148k_dfs', sep=',', header=None)

    bfs = [str(df.mean().round(2)[0]) + ' pm ' + str(df.std().round(2)[0]) for df in [gct_bfs_27, gct_bfs_148]]
    dfs = [str(df.mean().round(2)[0]) + ' pm ' + str(df.std().round(2)[0]) for df in [gct_dfs_27, gct_dfs_148]]

    cols = ['SHS27k', 'SHS148k']
    res3 = pd.DataFrame(columns=cols)
    rows = [combined.loc['bfs', 'X_all'].values, bfs, combined.loc['dfs', 'X_all'].values, dfs]
    for row in rows:
        res3 = res3.append({k: v for k, v in zip(cols, row)}, ignore_index=True)

    print(res3.to_latex())