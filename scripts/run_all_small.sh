bash scripts/aceso_gpt_search.sh small
bash scripts/aceso_gpt_execute.sh small
bash scripts/aceso_t5_search.sh small
bash scripts/aceso_t5_execute.sh small
bash scripts/aceso_resnet_search.sh small
bash scripts/aceso_resnet_execute.sh small

ray start --head
bash scripts/alpa_gpt_search_execute.sh small
bash scripts/alpa_wresnet_search_execute.sh small
ray stop

bash scripts/megatron_gpt_search.sh small
bash scripts/megatron_gpt_execute.sh small
bash scripts/megatron_t5_search.sh small
bash scripts/megatron_t5_execute.sh small
bash scripts/megatron_resnet_search.sh small
bash scripts/megatron_resnet_execute.sh small

python3 scripts/get_e2e_performance.py small
python3 scripts/get_search_cost.py small