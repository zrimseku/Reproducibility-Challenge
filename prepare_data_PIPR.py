import json
from gnn_data import GNN_DATA

if __name__ == '__main__':
    paths = {'SHS27k': "./data/protein.actions.SHS27k.STRING.txt",
             'SHS148k': "./data/protein.actions.SHS148k.STRING.txt",
             'STRING': "./data/9606.protein.actions.all_connected.txt"}
    for dataset in paths.keys():

        ppi_data = GNN_DATA(ppi_path=paths[dataset])

        interactions_with_labels = {'dict': ppi_data.ppi_dict, 'labels': ppi_data.ppi_label_list,
                                   'names': ppi_data.protein_name}

        jsobj = json.dumps(interactions_with_labels)
        with open(f'data/{dataset}_ppi_with_labels.json', 'w') as f:
            f.write(jsobj)