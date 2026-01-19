mkdir -p /mnt/hdfs/if_au/saves/mrx/merged

echo "MMAR"
bash auto_eval_mmar.sh
echo "MMAR 全部测完"

echo "MMAU_new"
bash auto_eval_mmau_new.sh
echo "MMAU_new 全部测完"

echo "MMAU_old"
bash auto_eval_mmau_old.sh
echo "MMAU_old 全部测完"