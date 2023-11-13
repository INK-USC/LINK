#!/usr/bin/env bash
set -x
set -e

model_name=("gpt")
order_name=("conclusion_first" "premise_first")
rule_path="../data/rules.json"


input_dir="path_to_conclusion_first_results"

save_dir="./LINK_llh"

rule_index="0 1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"

for name in "${model_name[@]}"; do
	python get_llh.py --input_dir $input_dir --model $name --rule_indexes $rule_index --output_dir $save_dir --meta_rule_info $rule_path
done



input_dir="path_to_premise_first_results"

save_dir="./LINK_llh"

rule_index="19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200"

for name in "${model_name[@]}"; do
	python get_llh.py --input_dir $input_dir --model $name --rule_indexes $rule_index --output_dir $save_dir --meta_rule_info $rule_path
done



for name in "${order_name[@]}"; do
	python concat_data.py --data_dir ${save_dir}/${name} --model_indexes gpt --primary_rank_model gpt
done
