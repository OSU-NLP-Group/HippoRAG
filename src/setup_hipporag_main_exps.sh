gpus_available=$1
syn_threshold=0.8

for data in musique 2wikimultihopqa hotpotqa;
do
  bash src/setup_hipporag.sh $data facebook/contriever gpt-3.5-turbo-1106 $gpus_available $syn_threshold
  bash src/setup_hipporag_colbert.sh $data gpt-3.5-turbo-1106 $gpus_available $syn_threshold
done