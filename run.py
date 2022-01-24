import os

def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs, seed):
    os.system("python -u gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            --seed={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs, seed))


if __name__ == "__main__":

    for split_mode in ["bfs", "dfs", "random"]:

        # set for training on different datasets
        dataset = "SSH27"             # select between SSH27k, SSH148k and STRING

        # set to False if you already have split data
        split_new = True

        # PPI network graph construction method, True: GCT, False: GCA (False is default except for table 5 experiments)
        graph_only_train = False

        # hyperparameters as described in the paper
        use_lr_scheduler = True
        batch_size = 1024
        epochs = 300

        save_path = "./save_model/"

        if dataset == 'SSH27k':
            ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
            pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
            vec_path = "./data/vec5_CTC.txt"

        elif dataset == 'SSH148k':
            ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
            pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
            vec_path = "./data/vec5_CTC.txt"

        elif dataset == 'STRING':
            ppi_path = "./data/9606.protein.actions.all_connected.txt"
            pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
            vec_path = "./data/vec5_CTC.txt"

        else:
            ValueError('Not a valid dataset!')

        for seed in [0, 42, 100, 600, 2000]:
            description = f"{dataset}_{split_mode}_{seed}"
            train_valid_index_path = f"new_train_valid_index_json/{description}.json"

            print('____________________________________________________________________________')
            print(f'____________________________{description}________________________________')
            print('____________________________________________________________________________')

            run_func(description, ppi_path, pseq_path, vec_path,
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train,
                    batch_size, epochs, seed)