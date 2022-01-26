from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Bidirectional, concatenate, multiply, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D, Input, CuDNNGRU
from tensorflow.python.keras.optimizers import Adam,  RMSprop
from tensorflow import keras

from PIPR.embeddings.seq2tensor import s2t
from utils import Metrictor_PPI


def get_session(gpu_fraction=0.75):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def prepare_data(id2seq_file, emb_file):
    seq2t = s2t(emb_file)
    pseq_dict = {}
    for line in tqdm(open(id2seq_file)):
        line = line.strip().split('\t')
        if line[0] not in pseq_dict.keys():
            pseq_dict[line[0]] = line[1]

    dim = seq2t.dim
    seq_tensor = np.array([seq2t.embed_normalized(pseq_dict[idx2name[i]], seq_size) for i in range(len(id2index))]).astype(dtype=np.float16)

    seq_index1 = []
    seq_index2 = []
    class_labels = []
    for interaction in ppi_with_labels['dict'].keys():
        p1, p2 = interaction.split('__')
        seq_index1.append(id2index[p1])
        seq_index2.append(id2index[p2])
        label_id = ppi_with_labels['dict'][interaction]
        class_labels.append(ppi_with_labels['labels'][label_id])

    seq_index1 = np.array(seq_index1)
    seq_index2 = np.array(seq_index2)
    class_labels = np.array(class_labels)
    return seq_tensor, seq_index1, seq_index2, class_labels, dim


def make_splits(dataset, mode, seeds, seq_index1, seq_index2):
    splits = []

    for seed in seeds:
        split_path = f'../../../../new_train_valid_index_json/{dataset}_{mode}_{seed}.json'

        with open(split_path, 'r') as f:
            split_dict = json.load(f)

        train_ids = split_dict['train_index']
        test_ids = split_dict['valid_index']

        counts = defaultdict(int)
        for i in train_ids:
            counts[seq_index1[i]] = 1
            counts[seq_index2[i]] = 1

        test_bs = []
        test_es = []
        test_ns = []
        for i in test_ids:
            seen = counts[seq_index1[i]] + counts[seq_index2[i]]
            if seen == 2:
                test_bs.append(i)
            elif seen == 1:
                test_es.append(i)
            elif seen == 0:
                test_ns.append(i)
            else:
                print(counts[i])

        splits.append((train_ids, test_bs, test_es, test_ns))

    return splits


def build_model(dim, hidden_dim=25):
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(2)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(2)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(2)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(2)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(2)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(2)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(hidden_dim, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(7, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model


if __name__ == '__main__':
    batch_size = 100
    learning_rate = 0.001

    dataset = 'SHS27k'

    # Note: if you use another PPI dataset, this needs to be changed to a corresponding dictionary file.
    id2seq_file = f"../../../../data/protein.{dataset}.sequences.dictionary.tsv"
    # id2seq_file = "../protein.sequences.dictionary.both.tsv"

    seq_size = 2000
    emb_file = "../../../../data/vec5_CTC.txt"
    n_epochs = 1


    # ds_file, label_index, rst_file, use_emb, hiddem_dim
    ds_file = f"../../../../data/protein.actions.{dataset}.STRING.txt"
    # ds_file = "../protein.actions.SHS27k.tsv"

    tf.keras.backend.set_session(get_session())

    with open(f'../../../../data/{dataset}_ppi_with_labels.json', 'r') as f:
        ppi_with_labels = json.load(f)

    id2index = ppi_with_labels['names']
    idx2name = {v: k for k, v in zip(id2index.keys(), id2index.values())}

    seq_tensor, seq_index1, seq_index2, class_labels, dim = prepare_data(id2seq_file, emb_file)

    mode = 'random'
    seeds = [0, 42]
    splits = make_splits(dataset, mode, seeds, seq_index1, seq_index2)

    for train, tbs, tes, tns in splits:

        # Training
        merge_model = build_model(dim)
        adam = Adam(lr=learning_rate, amsgrad=True)
        rms = RMSprop(lr=learning_rate)

        merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]]], class_labels[train], batch_size=batch_size, epochs=n_epochs)
        #result1 = merge_model.evaluate([seq_tensor1[test], seq_tensor2[test]], class_labels[test])

        # Testing
        results = []
        test_sets = [tbs + tes + tns, tbs, tes, tns]
        for i, test in enumerate(test_sets):
            if len(test) == 0:
                results.append(None)
                continue
            pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]]])

            pred_bool = pred > 0.5

            metrics = Metrictor_PPI(pred_bool, class_labels[test])
            metrics.show_result()

            # for i in range(len(class_labels[test])):
            #     num_total += 1
            #     if np.argmax(class_labels[test][i]) == np.argmax(pred[i]):
            #         num_hit += 1
            # accuracy = num_hit / num_total
            # print(accuracy)

            print(f"---------------- valid-test-{['all', 'bs', 'es', 'ns'][i]} result --------------------")

            print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))

            results.append(metrics.F1)

        with open(f'../../../../save_test_results/PIPR_{dataset}_{mode}', 'a') as f:
            f.write(", ".join(str(r) for r in results))
            f.write('\n')


