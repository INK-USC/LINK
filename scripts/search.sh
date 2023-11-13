rule_keys="rule0 rule1 rule2 rule3 rule4 rule5 rule6 rule7 rule8 rule9"
output_directory="../output/all_new_rules_seperation_v2/"
rule_path="../data/rules.json"

python ../LINK/knowledge_beam_search.py --do_search --knowledge_n_sample 50 --deduplicate --rule_keys $rule_keys --output_directory $output_directory --rule_path $rule_path --get_verifier_samples --factual_verifier_threshold 0.85 --datatype_verifier_threshold 0.85 --beam_size 200 --traverse_order premise --accumulate_verifier_confidence --dynamic_verifier_threshold --dynamic_ranker
