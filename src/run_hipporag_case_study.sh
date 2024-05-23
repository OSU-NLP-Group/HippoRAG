for data in books movies biomed university;
do
  #ColBERT Baseline
  python3 src/ircot_hipporag.py --dataset case_study_$data --retriever colbertv2 --max_steps 1 --dpr_only t --doc_ensemble f --top_k 10  --sim_threshold 0.8 --damping 0.5

  #IRCoT Baseline
  python3 src/ircot_hipporag.py --dataset case_study_$data --retriever colbertv2 --max_steps 4 --dpr_only t --doc_ensemble f --top_k 10  --sim_threshold 0.8 --damping 0.5

  #HippoRAG
  python3 src/ircot_hipporag.py --dataset case_study_$data --retriever colbertv2 --max_steps 1 --doc_ensemble f --top_k 10  --sim_threshold 0.8 --damping 0.5
done