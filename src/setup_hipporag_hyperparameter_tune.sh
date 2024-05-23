gpus_available=$1

for syn_threshold in 0.8 0.9;
do
  bash src/setup_hipporag.sh musique_train facebook/contriever gpt-3.5-turbo-1106 $gpus_available $syn_threshold
  bash src/setup_hipporag_colbert.sh musique_train gpt-3.5-turbo-1106 $gpus_available $syn_threshold
done