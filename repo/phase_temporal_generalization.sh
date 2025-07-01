SUBJECTS="EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EPc13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33"

cd /project_data/volume0/tpirnia/thesis_repository/repo/sig

t=0.55 #threshold

for s in $SUBJECTS; do 
python find_null_clusters.py $s; #<-- check this to change or confirm it can take $t as an additional arg
python find_clusters.py $s $t;
python find_sig_clusters.py $s $t;
python plot_corrected_tg.py $s $t;

python plot_null_cluster_distribution.py $s $t;

# for s in $SUBJECTS; do 
python find_loo_null_clusters.py $s $t;
python find_loo_clusters.py $s $t;
python find_loo_sig_clusters.py $s $t;
python plot_loo_corrected_tg.py $s $t;
done