for data in musique 2wikimultihopqa hotpotqa;
do
  #When max_steps is 1, this script does single step retrieval
  python3 ircot_hipporag.py --dataset $data --retriever colbertv2 --max_steps 1 --doc_ensemble f --top_k 10  --sim_threshold 0.8 --damping 0.5
done