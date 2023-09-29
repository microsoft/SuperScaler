
echo [TIME] before profiling: $(date '+%Y-%m-%d-%H-%M-%S') >> profile_log.log
python3 gen_prof_database.py --max-comm-size-intra-node 32 --max-comm-size-inter-node 29
echo [TIME] after profiling: $(date '+%Y-%m-%d-%H-%M-%S') >> profile_log.log