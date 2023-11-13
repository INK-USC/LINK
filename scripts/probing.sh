rule_keys="rule0 rule1 rule2 rule3 rule4 rule5 rule6 rule7 rule8 rule9"
positive_conclusion_rules="rule26 rule27 rule28 rule29 rule48 rule49 rule50 rule51 rule52 rule53 rule54 rule55 rule56 rule57 rule58 rule59 rule60 rule61 rule62 rule63 rule64 rule158 rule159 rule160 rule161 rule162 rule163 rule164 rule165 rule166 rule167 rule168 rule169 rule170 rule171 rule172 rule173 rule174 rule175 rule176 rule177 rule178 rule179 rule180 rule181 rule182 rule183 rule184 rule185 rule186 rule187 rule188 rule189 rule190 rule191 rule192 rule193 rule194 rule195 rule196 rule197 rule198 rule199 rule200"
probe_model="gpt-4"
output_directory="../output/probing_set"
rule_path="../data/rules.json"
method_name="LINK"

# with cot
python ../LINK/probing.py --output_directory $output_directory --rule_keys $rule_keys --do_probe --method_name $method_name --collect_rationale --rule_path $rule_path --traverse_order premise --probe_model_path $probe_model --cot --cot_icl --positive_conclusion_rules $positive_conclusion_rules

# without cot
python ../LINK/probing.py --output_directory $output_directory --rule_keys $rule_keys --do_probe --method_name $method_name --collect_rationale --rule_path $rule_path --traverse_order premise --probe_model_path $probe_model --positive_conclusion_rules $positive_conclusion_rules