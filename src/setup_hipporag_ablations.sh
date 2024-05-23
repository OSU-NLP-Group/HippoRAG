gpus_available=$1
syn_threshold=0.8

for data in musique 2wikimultihopqa hotpotqa;
do
  for model in meta-llama/Llama-3-8b-chat-hf meta-llama/Llama-3-70b-chat-hf rebel;
  do
    bash src/setup_hipporag_colbert.sh $data $model $gpus_available $syn_threshold
  done
done