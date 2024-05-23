echo ''
echo 'COLBERT HYPERPARAMETER TUNING'
echo ''

for damping in 0.1 0.3 0.5 0.9;
do
  for syn_thresh in 0.8 0.9;
  do
    python3 src/ircot_hipporag.py --dataset musique_train --retriever colbertv2 --max_steps 1 --doc_ensemble f --top_k 10  --sim_threshold $syn_thresh --damping $damping
  done
done

echo ''
echo 'CONTRIEVER HYPERPARAMETER TUNING'
echo ''

for damping in 0.1 0.3 0.5 0.9;
do
  for syn_thresh in 0.8 0.9;
  do
    python3 src/ircot_hipporag.py --dataset musique_train --retriever facebook/contriever --max_steps 1 --doc_ensemble f --top_k 10  --sim_threshold $syn_thresh --damping $damping
  done
done
