from argparse import ArgumentParser
from glob import glob
import os
import json
from collections import defaultdict as ddict
from glob import glob
import re



parser = ArgumentParser()
parser.add_argument("--data_dir")
# parser.add_argument("--rule_indexes",nargs="+",default=["1","3","4"])
parser.add_argument("--model_indexes",nargs="+",default=["ft","llama","gpt"])
parser.add_argument("--succint",action = "store_true",help="whether we are dealing with succint rules or not.\
		     If they are succint rules, then it should have a files indicating while rules we need from its\
		     corresponding full rules")

parser.add_argument("--top_beams",type=int,default=-1)
parser.add_argument("--primary_rank_model", default="llama")
args = parser.parse_args()


def what_node_len(key):
	key = key.split("_")[-1]
	assert isinstance(int(key),int)
	return key


to_be_plot_rules_path_template = f"{args.data_dir}"
output_name = "concat.json"
if args.succint:
	args.data_dir += "_succint"
output_path = os.path.join(args.data_dir,output_name)

if not os.path.exists(args.data_dir):
	print("We dont have {}".format(args.data_dir))
	exit()


model_data_d = ddict(lambda: ddict(list))

max_node_len_file = f"max_node_len.json"
max_node_len_path = os.path.join(args.data_dir,max_node_len_file)
with open(max_node_len_path) as f:
	max_node_len_d = json.load(f)

check_sorted_score_keys_d = ddict(list)
assert args.primary_rank_model in args.model_indexes, f"need {args.primary_rank_model} for ranking"
primary_rank_model_index = args.model_indexes.index(args.primary_rank_model)
args.model_indexes[primary_rank_model_index], args.model_indexes[0] = args.model_indexes[0], args.model_indexes[primary_rank_model_index]


for model_name in args.model_indexes:
	
	file_pattern = f"rule*_agreement_{model_name}.json"
	path = os.path.join(args.data_dir,file_pattern)
	all_available_rules = glob(path)
	for available_rule in all_available_rules:

		with open(available_rule) as f:
			scores = json.load(f)

		regex_pattern = re.compile(rf'.*rule(\d*)_agreement_{model_name}.json')
		m = re.match(regex_pattern,available_rule)
		true_rule = f"rule{m.group(1)}"

		
		sorted_score_keys = sorted(scores.keys())


		if args.top_beams != -1 and model_name == args.primary_rank_model and not args.succint:
			searched_scores = ddict(dict)


			for key in scores:
				if "tail" in key:
					is_tail_or_head = "tail"
				else:
					is_tail_or_head = "head"

				if int(key.rsplit('_',1)[-1]) == int(max_node_len_d[true_rule]):
					searched_scores[is_tail_or_head][key] = scores[key]


			for is_tail_or_head in searched_scores:
				be_keeped_rules = sorted(list(searched_scores[is_tail_or_head].keys()),key = lambda x: searched_scores[is_tail_or_head][x][args.primary_rank_model])
				be_keeped_rules = be_keeped_rules[:args.top_beams] if is_tail_or_head == "tail" else be_keeped_rules[-args.top_beams:]
				be_keeped_rules = [x.rsplit("_",1)[0] for x in be_keeped_rules]
				searched_scores[is_tail_or_head] = be_keeped_rules
			searched_scores = dict(searched_scores)
			to_be_plot_rules_path = os.path.join(to_be_plot_rules_path_template,f"{true_rule}_needed_for_plot.json")
			with open(to_be_plot_rules_path,"w") as f:
				json.dump(searched_scores,f)
		
		
		if args.top_beams != -1:
			try:
				to_be_plot_rules_path = os.path.join(to_be_plot_rules_path_template,f"{true_rule}_needed_for_plot.json")
				with open(to_be_plot_rules_path) as f:
					to_be_plot_rules = json.load(f)
			except:
				print("You must run full version of rules first")

		for key in sorted_score_keys:
			
			if "tail" in key:
				is_tail_or_head = "tail"
			else:
				is_tail_or_head = "head"

			if args.top_beams != -1:

				if not args.succint:
					if key.rsplit("_",1)[0] not in to_be_plot_rules[is_tail_or_head]:
						continue
				
				if args.succint:
					if key not in to_be_plot_rules[is_tail_or_head]:
						continue




			model_data_d[model_name][key].append(scores[key][model_name])
			if not args.succint:
				model_data_d[model_name]["all_tail_ruleall_nodelen_all" if is_tail_or_head == 'tail' else "all_head_ruleall_nodelen_all"].append(scores[key][model_name])
				model_data_d[model_name][f"all_tail_{true_rule}_nodelen_all" if is_tail_or_head == 'tail' else f"all_head_{true_rule}_nodelen_all"].append(scores[key][model_name])
				model_data_d[model_name][f"all_tail_{true_rule}_nodelen_{what_node_len(key)}" if is_tail_or_head == 'tail' else f"all_head_{true_rule}_nodelen_{what_node_len(key)}"].append(scores[key][model_name])
				model_data_d[model_name][f"all_tail_ruleall_nodelen_{what_node_len(key)}" if is_tail_or_head == 'tail' else f"all_head_ruleall_nodelen_{what_node_len(key)}"].append(scores[key][model_name])
				if int(what_node_len(key)) == int(max_node_len_d[true_rule]):
					model_data_d[model_name][f"all_tail_{true_rule}_nodelen_max" if is_tail_or_head == 'tail' else f"all_head_{true_rule}_nodelen_max"].append(scores[key][model_name])
					model_data_d[model_name][f"all_tail_ruleall_nodelen_max" if is_tail_or_head == 'tail' else f"all_head_ruleall_nodelen_max"].append(scores[key][model_name])
			else:
				model_data_d[model_name]["all_tail_ruleall_nodelen_max" if is_tail_or_head == 'tail' else "all_head_ruleall_nodelen_max"].append(scores[key][model_name])
				model_data_d[model_name][f"all_tail_{true_rule}_nodelen_max" if is_tail_or_head == 'tail' else f"all_head_{true_rule}_nodelen_max"].append(scores[key][model_name])
				




with open(output_path,"w") as f:
	json.dump(model_data_d,f)

