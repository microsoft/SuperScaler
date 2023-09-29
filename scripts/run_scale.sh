bash scripts/aceso_gpt_search.sh scale
bash scripts/aceso_gpt_execute.sh scale

ray start --head
bash scripts/alpa_gpt_search_execute.sh scale
ray stop

bash scripts/megatron_gpt_search.sh scale
bash scripts/megatron_gpt_execute.sh scale