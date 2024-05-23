data=$1
retriever_name=$2
extraction_model=$3
available_gpus=$4
syn_thresh=$5
extraction_type=ner

#Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --model_name $extraction_model --run_ner --num_passages all #MuSiQue NER
python src/named_entity_extraction_parallel.py --dataset $data --model_name $extraction_model

#Creating Contriever Graph
python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --extraction_type $extraction_type --cosine_sim_edges

#Getting Nearest Neighbor Files
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/query_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/kb_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/rel_kb_to_kb.tsv

python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --create_graph --extraction_type $extraction_type --cosine_sim_edges