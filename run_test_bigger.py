import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, bigger_ppi_path, bigger_pseq_path):
    os.system("python gnn_test_bigger.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --bigger_ppi_path={} \
            --bigger_pseq_path={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, bigger_ppi_path, bigger_pseq_path))


if __name__ == "__main__":

    seeds = [0, 42, 100, 600, 2000]
    for split_mode in ["bfs", "dfs", "random"]:
        for seed in seeds:

            # set for training on different datasets
            dataset = "SSH27k"  # select between SSH27k, SSH148k

            save_model_path = "./save_model"

            if dataset == 'SSH27k':
                ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
                pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
                vec_path = "./data/vec5_CTC.txt"

            elif dataset == 'SSH148k':
                ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
                pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
                vec_path = "./data/vec5_CTC.txt"

            else:
                raise ValueError(f'{dataset} is not a valid dataset!')

            description = f"{dataset}_{split_mode}_{seed}"
            index_path = f"new_train_valid_index_json/{description}.json"
            gnn_model = os.path.join(save_model_path, f"gnn_{description}", "gnn_model_valid_best.ckpt")

            bigger_ppi_path = "./data/9606.protein.actions.all_connected.txt"
            bigger_pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"

            run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, bigger_ppi_path,
                     bigger_pseq_path)
