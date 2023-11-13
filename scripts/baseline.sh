output_directory="../output/baseline_no_logit_bias/"
rule_keys="rule0 rule1 rule2 rule3 rule4 rule5 rule6 rule7 rule8 rule9"
rule_path="../data/rules.json"
model="gpt-3.5-turbo-16k"

python ../LINK/baseline.py --beam_size 200 --search_n_sample 50 --do_search --deduplicate --full --output_directory $output_directory --rule_keys $rule_keys --knowledge_model_path $model --meta_rule_info $rule_path

