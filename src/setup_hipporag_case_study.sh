gpus_available=$1
syn_threshold=0.8

for data in books movies biomed university;
do
  bash src/setup_hipporag_colbert.sh case_study_$data gpt-3.5-turbo-1106 $gpus_available $syn_threshold
done