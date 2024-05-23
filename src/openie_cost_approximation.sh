#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-8b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-8b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-8b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-8b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-70b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-70b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name 2wikimultihopqa --run_ner --num_examples 6119 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-70b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py 2wikimultihopqa meta-llama/Llama-3-70b-chat-hf

#python named_entity_extraction_parallel.py hotpotqa #HotpotQA

#python openie_with_retrieval_option_parallel.py hippoqa_books_filtered True False False False False False False 1166 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_books_filtered
#
#python openie_with_retrieval_option_parallel.py hippoqa_movie_filtered True False False False False False False 1102 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_movie_filtered
#
#python openie_with_retrieval_option_parallel.py hippoqa_biomed_filtered True False False False False False False 897 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_biomed_filtered

#python openie_with_retrieval_option_parallel.py hippoqa_academic True False False False False False False 25 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_academic

#python openie_with_retrieval_option_parallel.py hippoqa_university True False False False False False False 161 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_university

#python openie_with_retrieval_option_parallel.py hippoqa_synthetic True False False False False False False 200 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_synthetic

#python openie_with_retrieval_option_parallel.py hippoqa_books True False False False False False False 5242 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_books
#
#python openie_with_retrieval_option_parallel.py hippoqa_movie True False False False False False False 5891 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_movie
#
#python openie_with_retrieval_option_parallel.py hippoqa_biomed True False False False False False False 3711 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py hippoqa_biomed

#python openie_with_retrieval_option_parallel.py musique True False False False False False False 100 0.6 #Goodreads NER
#python openie_with_retrieval_option_parallel.py musique True False False False False False False 11656 0.6 #Goodreads NER
#python named_entity_extraction_parallel.py musique

#python openie_with_retrieval_option_parallel.py musique True False False False False False 100 0.6 #MuSiQue NER
#python openie_with_retrieval_option_parallel.py musique True False False False False False 19990 0.6 #MuSiQue NER
#python named_entity_extraction_parallel.py musique

#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-8b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-8b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-8b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-8b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name musique --run_ner --num_examples 11656 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-70b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py musique meta-llama/Llama-3-70b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name 2wikimultihopqa --run_ner --num_examples 6119 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-70b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py 2wikimultihopqa meta-llama/Llama-3-70b-chat-hf

#python openie_with_retrieval_option_parallel.py --dataset_name hotpotqa --run_ner --num_examples 9221 --synonym_threshold 0.6 --model_name meta-llama/Llama-3-70b-chat-hf #MuSiQue NER
#python named_entity_extraction_parallel.py hotpotqa meta-llama/Llama-3-70b-chat-hf

#python openie_with_retrieval_option_parallel.py 2wikimultihopqa True False False False False False False 100 0.6 #2WikiMultiHop NER
#python openie_with_retrieval_option_parallel.py 2wikimultihopqa True False False False False False False 6119 0.6 #2WikiMultiHop NER
#python named_entity_extraction_parallel.py 2wikimultihopqa #2WikiMultiHop

#Ablations
#python openie_with_retrieval_option.py True False False False False False 9221 0.6
#python openie_with_retrieval_option.py True True False False False False 9221 0.6