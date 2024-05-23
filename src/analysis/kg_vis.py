import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', type=str,
                        default='output/musique_facts_and_sim_graph_relation_dict_ents_only_lower_preprocess_ner_colbertv2.v3.subset.p')
    args = parser.parse_args()

    kg = pickle.load(open(args.kg, 'rb'))
    print(len(kg))

    res = []
    for head_tail in kg:
        if head_tail[0].lower() in ['alhandra', 'vila franca de xira'] or head_tail[1].lower() in ['alhandra', 'vila franca de xira']:
            head = head_tail[0]
            tail = head_tail[1]
            relation = kg[head_tail]
            res.append((head, relation, tail))

    print(res)
