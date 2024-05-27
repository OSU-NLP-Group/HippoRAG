import _pickle as pickle
import argparse
from glob import glob
import os.path

import pandas as pd

import pickle
import numpy as np
import os
from tqdm import tqdm
import torch

import faiss
import gc

from transformers import AutoModel, AutoTokenizer
from processing import *

# TODO: Change hard-coded vector output directory
VECTOR_DIR = 'data/lm_vectors'


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class RetrievalModule:
    """
    Class designed to retrieve potential synonymy candidates for a set of UMLS terms from a set of entities.
    """

    def __init__(self,
                 retriever_name,
                 string_filename,
                 pool_method='cls'
                 ):
        """
        Args:
            retriever_name: Retrieval names can be one of 3 types
                2) The name of a pickle file mapping AUIs to precomputed vectors
                3) A huggingface transformer model
        """

        self.retriever_name = retriever_name

        self.retrieval_name_dir = None

        # Search for pickle file
        print('No Pre-Computed Vectors. Confirming PLM Model.')

        try:
            if 'ckpt' in retriever_name:
                self.plm = AutoModel.load_from_checkpoint(retriever_name)
            else:
                self.plm = AutoModel.from_pretrained(retriever_name)
        except:
            assert False, print('{} is an invalid retriever name. Check Documentation.'.format(retriever_name))

        # If not pre-computed, create vectors
        self.retrieval_name_dir = VECTOR_DIR + '/' + self.retriever_name.replace('/', '_').replace('.', '') + '_' + pool_method

        if not (os.path.exists(self.retrieval_name_dir)):
            os.makedirs(self.retrieval_name_dir)

        # Get previously computed vectors
        precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors(self.retrieval_name_dir)

        # Get AUI Strings to be Encoded
        string_df = pd.read_csv(string_filename, sep='\t')
        string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
        sorted_df = self.create_sorted_df(string_df.strings.values)

        # Identify Missing Strings
        missing_strings = self.find_missing_strings(sorted_df.strings.unique(), precomp_strings)

        # Encode Missing Strings
        if len(missing_strings) > 0:
            print('Encoding {} Missing Strings'.format(len(missing_strings)))
            new_vectors, new_strings, = self.encode_strings(missing_strings, pool_method)

            precomp_strings = list(precomp_strings)
            precomp_vectors = list(precomp_vectors)

            precomp_strings.extend(list(new_strings))
            precomp_vectors.extend(list(new_vectors))

            precomp_vectors = np.array(precomp_vectors)

            self.save_vecs(precomp_strings, precomp_vectors, self.retrieval_name_dir)

        self.vector_dict = self.make_dictionary(sorted_df, precomp_strings, precomp_vectors)

        print('Vectors Loaded.')

        queries = string_df[string_df.type == 'query']
        kb = string_df[string_df.type == 'kb']

        nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values)
        pickle.dump(nearest_neighbors, open(self.retrieval_name_dir + '/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0]), 'wb'))

    def get_precomputed_plm_vectors(self, retrieval_name_dir):

        # Load or Create a DataFrame sorted by phrase length for efficient PLM computation
        strings = self.load_precomp_strings(retrieval_name_dir)
        vectors = self.load_plm_vectors(retrieval_name_dir)

        return strings, vectors

    def create_sorted_df(self, strings):
        lengths = []

        for string in tqdm(strings):
            lengths.append(len(str(string)))

        lengths_df = pd.DataFrame(lengths)
        lengths_df['strings'] = strings

        return lengths_df.sort_values(0)

    def save_vecs(self, strings, vectors, direc_name, bin_size=50000):

        with open(direc_name + '/encoded_strings.txt', 'w') as f:
            for string in strings:
                f.write(string + '\n')

        split_vecs = np.array_split(vectors, int(len(vectors) / bin_size) + 1)

        for i, vecs in tqdm(enumerate(split_vecs)):
            pickle.dump(vecs, open(direc_name + '/vecs_{}.p'.format(i), 'wb'))

    def load_precomp_strings(self, retrieval_name_dir):
        filename = retrieval_name_dir + '/encoded_strings.txt'

        if not (os.path.exists(filename)):
            return []

        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

        return lines

    def load_plm_vectors(self, retrieval_name_dir):
        vectors = []

        print('Loading PLM Vectors.')
        files = glob(retrieval_name_dir + '/vecs_*.p')

        if len(files) == 0:
            return vectors

        for i in tqdm(range(len(files))):
            i_files = glob(retrieval_name_dir + '/*_{}.p'.format(i))
            if len(i_files) != 1:
                break
            else:
                vectors.append(pickle.load(open(i_files[0], 'rb')))

        vectors = np.vstack(vectors)

        return vectors

    def find_missing_strings(self, relevant_strings, precomputed_strings):

        return list(set(relevant_strings).difference(set(precomputed_strings)))

    def make_dictionary(self, sorted_df, precomp_strings, precomp_vectors):

        print('Populating Vector Dict')
        precomp_string_ids = {}

        for i, string in enumerate(precomp_strings):
            precomp_string_ids[string] = i

        vector_dict = {}

        for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
            string = row.strings

            try:
                vector_id = precomp_string_ids[string]
                vector_dict[string] = precomp_vectors[vector_id]
            except:
                ipdb.set_trace()

        return vector_dict

    def encode_strings(self, strs_to_encode, pool_method):
        self.plm.to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)

        # Sorting Strings by length
        sorted_missing_strings = [len(s) for s in strs_to_encode]
        strs_to_encode = list(np.array(strs_to_encode)[np.argsort(sorted_missing_strings)])

        all_cls = []
        all_strings = []
        num_strings_proc = 0

        with torch.no_grad():

            batch_sizes = []

            text_batch = []
            max_pad_size = 0

            for i, string in tqdm(enumerate(strs_to_encode), total=len(strs_to_encode)):

                length = len(tokenizer.tokenize(string))

                text_batch.append(string)
                num_strings_proc += 1

                if length > max_pad_size:
                    max_pad_size = length

                if max_pad_size * len(text_batch) > 50000 or num_strings_proc == len(strs_to_encode):

                    text_batch = list(text_batch)
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.plm.config.max_length)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    input_ids = input_ids.to('cuda')
                    attention_mask = attention_mask.to('cuda')

                    outputs = self.plm(input_ids, attention_mask=attention_mask)

                    if pool_method == 'cls':
                        embeddings = outputs[0][:, 0, :]

                    elif pool_method == 'mean':
                        embeddings = mean_pooling(outputs[0], attention_mask)

                    all_cls.append(embeddings.cpu().numpy())
                    all_strings.extend(text_batch)

                    batch_sizes.append(len(text_batch))

                    text_batch = []
                    max_pad_size = 0

        all_cls = np.vstack(all_cls)

        assert len(all_cls) == len(all_strings)
        assert all([all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))])

        return all_cls, all_strings

    def retrieve_knn(self, queries, knowledge_base, k=2047):

        original_vecs = []
        new_vecs = []

        for string in knowledge_base:
            original_vecs.append(self.vector_dict[string])

        for string in queries:
            new_vecs.append(self.vector_dict[string])

        if len(original_vecs) == 0 or len(new_vecs) == 0:
            return {}

        original_vecs = np.vstack(original_vecs)
        new_vecs = np.vstack(new_vecs)

        original_vecs = original_vecs.astype(np.float32)
        new_vecs = new_vecs.astype(np.float32)

        faiss.normalize_L2(original_vecs)
        faiss.normalize_L2(new_vecs)

        # Preparing Data for k-NN Algorithm
        print('Chunking')

        dim = len(original_vecs[0])
        index_split = 4
        index_chunks = np.array_split(original_vecs, index_split)
        query_chunks = np.array_split(new_vecs, 100)

        # Building and Querying FAISS Index by parts to keep memory usage manageable.
        print('Building Index')

        index_chunk_D = []
        index_chunk_I = []

        current_zero_index = 0

        for num, index_chunk in enumerate(index_chunks):

            print('Running Index Part {}'.format(num))
            index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)  # build the index

            if faiss.get_num_gpus() > 1:
                gpu_resources = []

                for i in range(faiss.get_num_gpus()):
                    res = faiss.StandardGpuResources()
                    gpu_resources.append(res)

                gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
            else:
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

            print()
            gpu_index.add(index_chunk)

            D, I = [], []

            for q in tqdm(query_chunks):
                d, i = gpu_index.search(q, k)

                i += current_zero_index

                D.append(d)
                I.append(i)

            index_chunk_D.append(D)
            index_chunk_I.append(I)

            current_zero_index += len(index_chunk)

            #             print(subprocess.check_output(['nvidia-smi']))

            del gpu_index
            del gpu_resources
            gc.collect()

        print('Combining Index Chunks')

        stacked_D = []
        stacked_I = []

        for D, I in zip(index_chunk_D, index_chunk_I):
            D = np.vstack(D)
            I = np.vstack(I)

            stacked_D.append(D)
            stacked_I.append(I)

        del index_chunk_D
        del index_chunk_I
        gc.collect()

        stacked_D = np.hstack(stacked_D)
        stacked_I = np.hstack(stacked_I)

        full_sort_I = []
        full_sort_D = []

        for d, i in tqdm(zip(stacked_D, stacked_I)):
            sort_indices = np.argsort(d, kind='stable')

            sort_indices = sort_indices[::-1]

            i = i[sort_indices][:k]
            d = d[sort_indices][:k]

            full_sort_I.append(i)
            full_sort_D.append(d)

        del stacked_D
        del stacked_I
        gc.collect()

        sorted_candidate_dictionary = {}

        for new_index, nn_info in tqdm(enumerate(zip(full_sort_I, full_sort_D))):
            nn_inds, nn_dists = nn_info
            nns = [knowledge_base[i] for i in nn_inds]

            sorted_candidate_dictionary[queries[new_index]] = (nns, nn_dists)

        return sorted_candidate_dictionary


import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_name', type=str)
    parser.add_argument('--string_filename', type=str)
    parser.add_argument('--pool_method', type=str, default='mean')

    args = parser.parse_args()

    retriever_name = args.retriever_name
    string_filename = args.string_filename
    pool_method = args.pool_method

    retrieval_module = RetrievalModule(retriever_name, string_filename, pool_method)
