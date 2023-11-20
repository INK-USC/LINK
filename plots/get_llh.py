import os
from copy import deepcopy
import sys

sys.path.append("../link/")
from knowledge_beam_search import get_rule,_get_full_rule_predicates,_swap_conclusion_to_last
import json
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM,LlamaTokenizer,LlamaForCausalLM
import glob
import csv
from collections import defaultdict as ddict
from numpy.linalg import norm
from numpy import dot
import numpy as np
from itertools import combinations
import backoff
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
import pathlib


class MyError(Exception):
    def __init__(self, message = "Path Error"):
        self.message = message
        super().__init__(self.message)

class Handler:
	def __init__(self,path,args,output_dir, rank_as_pre_con,traverse_order_entity_force):
		self.args = args
		self.traverse_order_entity_force = traverse_order_entity_force
		self.max_node_for_rule = 0
		self.path = path
		self._which_rule = self.which_rule()
		self.search_order = self.decide_order()
		output_dir = output_dir.format(search_order = self.search_order)
		pathlib.Path(output_dir).mkdir(exist_ok= True, parents= True)
		self.output_dir = output_dir

		self.plot_as_pre_con = rank_as_pre_con
		assert self._which_rule.startswith("rule"),"parse error"

		# self.conclusion_first_map = {
		# 	'rule1':['is_not_able_to_use(P, C)', 'was_invented_in(C, Y)', 'lived_during(P, D)', 'is_more_than_a_century_earlier_than(D, Y)', 'lived_in(P, A)', 'existed_during(A, D)'],
		# 	'rule3':['is_not_able_to_use(P, A)', 'was_invented_in(A, Y)', 'was_born_in(P, B)', 'is_more_than_a_century_earlier_than(B, Y)'],
		# 	'rule4':['is_not_able_to_eat(P, B)', 'is_ingredient_in(Z, B)', 'is_allergic_to(P, A)', 'is_one_type_of(Z, A)'],
		# 	'rule5':['not_drive(X, Y)', 'license_requirement(Y, B)', 'age_of(X, A)', 'smaller_than(A, B)'],
		# 	'rule6':['not_drive(X, Y)', 'issued_after(Y, B)', 'expired_license(X, A)', 'earlier_than(A, B)'],
		# 	'rule7':['not_drive(X, Y)', 'located_in(X, A)', 'not_allowed_by_city(Y, A)'],
		# 	'rule8':['not_play(X, Y)', 'not_available_in(Y, B)', 'resides_in(X, A)', 'located_in(A, B)'],
		# 	'rule9':['not_write(X, Y)', 'not_interested_in(X, A)', 'genre_of(Y, A)'],
		# 	'rule10':['not_write(X, Y)', 'published_in(Y, A)', 'born_in(X, B)', 'earlier_than(A, B)'],
		# 	'rule11':['not_create(X, Y)', 'lived_during(X, A)', 'not_from_time_period(Y, A)'],
		# 	'rule12':['not_watch(X, Y)', 'banned_in(Y, B)', 'resides_in(X, A)', 'located_in(A, B)'],
		# 	'rule13':['not_drink(X, Y)', 'religion_of(X, A)', 'not_allowed_by_religion(Y, A)'],
		# 	'rule14':['not_drink(X, Y)', 'allergic_to(X, Z)', 'contains(Y, Z)'],
		# 	'rule15':['not_eat(X, Y)', 'religion_of(X, A)', 'not_allowed_by_religion(Y, A)'],
		# 	'rule16':['not_plant(X, Y)', 'not_suitable_climate(Y, B)', 'located_in(X, A)', 'located_in(B, A)'],
		# 	'rule17':['not_wear(X, Y)', 'climate_of(X, A)', 'not_suitable_for(A, Y)'],
		# 	'rule18':['not_wear(X, Y)', 'climate_of(X, A)', 'not_suitable_for(A, Y)'],
		# 	'rule19':['not_sing(X, Y)', 'not_familiar_with(X, A)', 'language_of(Y, A)'],
		# 	'rule20':['not_attend(X, Y)', 'age_requirement(Y, B)', 'younger_than(X, A)', 'smaller_than(A, B)'],
		# 	'rule21':['not_bake(X, Y)', 'is_not_skillful_in(X, A)', 'required_for(Y, A)'],
		# 	'rule22':['not_celebrate(X, Y)', 'religion_of(X, A)', 'not_celebrated_by_religion(Y, A)'],
		# 	'rule23':['not_hunt(X, Y)', 'located_in(X, A)', 'not_found_in(Y, A)'],
		# 	'rule24':['not_harvest(X, Y)', 'lack_of_equipment(X, Z)', 'necessary_for_harvesting(Z, Y)']
		# }
		# self.premise_first_map = {
		# 	'rule4':['is_allergic_to(P, A)', 'is_one_type_of(Z, A)', 'is_not_able_to_eat(P, B)', 'is_ingredient_in(Z, B)'],
		# 	'rule7':['located_in(X, A)', 'not_drive(X, Y)', 'not_allowed_by_city(Y, A)'],
		# 	'rule8':['resides_in(X, A)', 'located_in(A, B)', 'not_play(X, Y)', 'not_available_in(Y, B)'],
		# 	'rule9':['not_interested_in(X, A)', 'not_write(X, Y)', 'genre_of(Y, A)'],
		# 	'rule11':['lived_during(X, A)', 'not_create(X, Y)', 'not_from_time_period(Y, A)'],
		# 	'rule12':['resides_in(X, A)', 'located_in(A, B)', 'not_watch(X, Y)', 'banned_in(Y, B)'],
		# 	'rule13':['religion_of(X, A)', 'not_drink(X, Y)', 'not_allowed_by_religion(Y, A)'],
		# 	'rule14':['allergic_to(X, Z)', 'not_drink(X, Y)', 'contains(Y, Z)'],
		# 	'rule15':['religion_of(X, A)', 'not_eat(X, Y)', 'not_allowed_by_religion(Y, A)'],
		# 	'rule16':['located_in(X, A)', 'located_in(B, A)', 'not_plant(X, Y)', 'not_suitable_climate(Y, B)'],
		# 	'rule17':['climate_of(X, A)', 'not_wear(X, Y)', 'not_suitable_for(A, Y)'],
		# 	'rule18':['climate_of(X, A)', 'not_wear(X, Y)', 'not_suitable_for(A, Y)'],
		# 	'rule19':['not_familiar_with(X, A)', 'not_sing(X, Y)', 'language_of(Y, A)'],
		# 	'rule21':['is_not_skillful_in(X, A)', 'not_bake(X, Y)', 'required_for(Y, A)'],
		# 	'rule22':['religion_of(X, A)', 'not_celebrate(X, Y)', 'not_celebrated_by_religion(Y, A)'],
		# 	'rule23':['located_in(X, A)', 'not_hunt(X, Y)', 'not_found_in(Y, A)'],
		# 	'rule24':['lack_of_equipment(X, Z)', 'not_harvest(X, Y)', 'necessary_for_harvesting(Z, Y)']
		# }


		# self.succint_rules = {
		# 	"rule1":["[P] lives in [A] during [D]","[P] is not able to use [C]"],
		# 	"rule3":["[P] is born in [B]","[P] is not able to use [A]"],
		# 	"rule4":["[P] is allergic to [A]","[P] is not able to eat [B]"],
		# }

		self.partial_rule_map = {}
		self.partial_word_map = {}
		self.convert_into_partial_rules("tail")
		self.convert_into_partial_rules("head")
		self.word_path = f"{self.output_dir}/{self._which_rule}_words.json"
		self.rule_path = f"{self.output_dir}/{self._which_rule}_rules.json"
		
		if not os.path.exists(self.rule_path):
			with open(f"{self.rule_path}","w") as f:
				json.dump(self.partial_rule_map,f)

		if not os.path.exists(self.word_path):
			with open(f"{self.word_path}","w") as f:
				json.dump(self.partial_word_map,f)

		

	def which_rule(self):
		return self.path["head"].split("/")[-1].split("_")[0]
	
	def read_file(self,path):
		try:
			with open(path) as f:
				reader = csv.reader(f)
				self.all_values = []
				for i,line in enumerate(reader):
					if i == 0:
						self.all_nodes = line
					else:
						self.all_values.append(line)
		except:
			raise MyError()
		

	def convert_into_partial_rules(self,tail_or_head):
		# for key in value_dict:
		# 	print("key: ", key, "value: ", value_dict[key])
		# 	premise = premise.replace("["+key+"]", " ".join(value_dict[key].split("_")))
		# 	conclusion = conclusion.replace("["+key+"]", " ".join(value_dict[key].split("_")))

		if tail_or_head == "head":
			path = self.path["head"]
		elif tail_or_head == "tail":
			path = self.path["tail"]
		_partial_rule_map = {}
		_partial_word_map = {}

		self.read_file(path)
		if not self.args.succint:

			partial_rules_id_temp = "{_which_rule}_{tail_or_head}_{rule_index}_{node_len}"
			for j,all_value in enumerate(self.all_values):
				all_value = dict(zip(self.all_nodes,all_value))
				# for i in range(2,len(self.all_nodes)+1):
				for i in range(len(self.all_nodes),len(self.all_nodes)+1):	
					all_node = self.all_nodes[:i]
					partial_rule = self._verbalize_with_partial_rules(all_node,all_value)
					partial_rules_id = partial_rules_id_temp.format(rule_index = j,node_len = i,_which_rule = self._which_rule, tail_or_head = tail_or_head)
					_partial_rule_map[partial_rules_id] = partial_rule
					_partial_word_map[partial_rules_id] = [all_value[_] for _ in all_node]

				self.max_node_for_rule = i if i > self.max_node_for_rule else self.max_node_for_rule

		else:
			partial_rules_id_temp = "{_which_rule}_{tail_or_head}_{rule_index}"
			*_, premise_template,conclusion_template = get_rule(self._which_rule,rule_file= args.meta_rule_info)
			# *_, premise_template,conclusion_template = self.succint_rules[self._which_rule]
			for j,all_value in enumerate(self.all_values):
				all_value = dict(zip(self.all_nodes,all_value))
				premise = premise_template
				conclusion = conclusion_template

				used_nodes = set()
				for key in all_value:
					if "["+key+"]" in premise_template or "["+key+"]" in conclusion_template:
						used_nodes.add(key)

					premise = premise.replace("["+key+"]", " ".join(all_value[key].split("_")))
					conclusion = conclusion.replace("["+key+"]", " ".join(all_value[key].split("_")))

				partial_rules_id = partial_rules_id_temp.format(rule_index = j,_which_rule = self._which_rule, tail_or_head = tail_or_head)
				_partial_rule_map[partial_rules_id] = premise + " " + conclusion
				_partial_word_map[partial_rules_id] = [all_value[_] for _ in used_nodes]



		self.partial_rule_map[tail_or_head] = _partial_rule_map
		self.partial_word_map[tail_or_head] = _partial_word_map

				

	def _verbalize_with_partial_rules(self,nodes,values):
		rule, graph, conclusion_first_order, premise_first_order, premise_template, conclusion_template = get_rule(self._which_rule,rule_file= args.meta_rule_info)
		if self.traverse_order_entity_force == "":
			if any([1 if _[1].nodetype == "factual" else 0 for _ in rule.variables.items()]):
				traverse_order = conclusion_first_order
				traverse_order_entity = "conclusion"
			else:
				traverse_order = premise_first_order
				traverse_order_entity = "premise"
		else:
			if self.traverse_order_entity_force == "conclusion":
				traverse_order = conclusion_first_order
				traverse_order_entity = "conclusion"
			elif self.traverse_order_entity_force == "premise":
				traverse_order = premise_first_order
				traverse_order_entity = "premise"


				
		full_rull_predicates = _get_full_rule_predicates(traverse_order,rule,graph, traverse_order_entity)
		if self.plot_as_pre_con:
			full_rull_predicates = _swap_conclusion_to_last(full_rull_predicates,list(rule.conclusion_map.keys())[0])
		output = rule.verbalize_with_partial_rules(full_rull_predicates,values,nodes)
		return output



	def decide_order(self):
		rule, graph, conclusion_first_order, premise_first_order, premise_template, conclusion_template = get_rule(self._which_rule,rule_file= args.meta_rule_info)
		if self.traverse_order_entity_force == "":
			if any([1 if _[1].nodetype == "factual" else 0 for _ in rule.variables.items()]):
				traverse_order_entity = "conclusion"
			else:
				traverse_order_entity = "premise"

			return traverse_order_entity
		else:
			return self.traverse_order_entity_force



class FT_model:
	def __init__(self):

		import fasttext
		self.ft = fasttext.load_model('path_to_fasttext/fasttext/crawl-300d-2M-subword.bin')

	def get_logprobs(self,inputs):
		scores = []
		for input in inputs:
			_input = combinations(input,2)
			cos_score_l = []
			for a,b in _input:
				a = self.ft[a]
				b = self.ft[b]
				cos_score = dot(a, b)/(norm(a)*norm(b))
				cos_score_l.append(cos_score)
			scores.append(np.mean(cos_score_l))
		return scores

def batch_process(l):
    for i in range(0,len(l),128):
        yield l[i:i+128]


class Probe_model:
	def __init__(self,model_path):
		# model_path = "decapoda-research/llama-7b-hf"
			
		DEFAULT_PAD_TOKEN = "[PAD]"
		DEFAULT_EOS_TOKEN = "</s>"
		DEFAULT_BOS_TOKEN = "<s>"
		DEFAULT_UNK_TOKEN = "</s>"
		self.tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side = "left")
		self.tokenizer.add_special_tokens(
			{
				"eos_token": DEFAULT_EOS_TOKEN,
				"bos_token": DEFAULT_BOS_TOKEN,
				"unk_token": DEFAULT_UNK_TOKEN,
				"pad_token": DEFAULT_PAD_TOKEN,
			})
		self.model = LlamaForCausalLM.from_pretrained(model_path,device_map = "auto",torch_dtype = torch.float16)

	@torch.no_grad()
	def get_logprobs(self,inputs):
		scores_l = []
		for batch in batch_process(inputs):
			input_texts = batch
			input_ids = self.tokenizer(input_texts, padding=True, return_tensors="pt").to('cuda')
			outputs = self.model(**input_ids)
			probs = torch.log_softmax(outputs.logits, dim=-1).detach()
			probs = probs[:, :-1, :]
			_input_ids = input_ids.input_ids[:, 1:]
			attention_mask = input_ids.attention_mask[:,1:]
			gen_probs = torch.gather(probs, 2, _input_ids[:, :, None]).squeeze(-1)
			gen_probs = gen_probs * attention_mask
			scores = torch.sum(gen_probs,dim = -1)
			scores = scores/ torch.sum(attention_mask,dim = -1)
			scores = scores.tolist()
			scores_l.extend(scores)
		return scores_l


class OpenAIWrapper:
	def __init__(self, model_path):
		self.model_path = model_path

	@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError))
	def get_logprobs(self,prompts,max_tokens=0):
		scores = []
		for prompt in prompts:
			response = openai.Completion.create(
				engine=self.model_path,
				prompt=prompt,
				temperature=0,
				top_p = 0,
				max_tokens=max_tokens,
				echo = True,
				logprobs=1
			)

			# first one is null
			score = np.mean(response["choices"][0]["logprobs"]['token_logprobs'][1:])
			scores.append(score)
		return scores


def score(model_name,model,head_or_tail,all_datas,bs = 64):
	
	keys_l = []
	scores_l = []
	if model_name != "ft":
		all_datas = all_datas[head_or_tail]["rules"]
	else:
		all_datas = all_datas[head_or_tail]["words"]

	datas = list(all_datas.values())
	keys = list(all_datas.keys())
	
	for i in range(0,len(datas),bs):
		data = datas[i:i+bs]
		key = keys[i:i+bs]
		scores = model.get_logprobs(data)
		scores = [np.float64(_) for _ in scores]
		keys_l.extend(key)
		scores_l.extend(scores)
	
	scores_l = [{f"{model_name}":_} for _ in scores_l]
	return dict(zip(keys_l,scores_l))


def pairwise_agreement(datas,model1,model2):
	all_keys = list(datas)
	agreements = []
	for pair0,pair1 in combinations(all_keys,2):
		if (datas[pair0][model1] > datas[pair0][model2] and datas[pair0][model2] > datas[pair1][model2]) or (datas[pair0][model1] < datas[pair0][model2] and datas[pair0][model2] < datas[pair1][model2]):
			agreements.append(1)
		else:
			agreements.append(0)
	return agreements


		

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",choices=["ft","llama","gpt","gpt_turbo"])
parser.add_argument("--rule_indexes",nargs="+")
parser.add_argument("--output_dir")
parser.add_argument("--traverse_order_entity_force",default="",type = str)
parser.add_argument("--input_dir")
parser.add_argument("--succint",action = "store_true")
parser.add_argument("--plot_as_pre_con",action="store_true",help="When we change the rank method during generation, we should keep the \
		    plots predicates order in line with what we used during the generation")
parser.add_argument("--meta_rule_info",type=str, default=None)

args = parser.parse_args()

rule_indexes = args.rule_indexes
path = args.input_dir
output_dir = args.output_dir
output_dir = os.path.join(output_dir,"{search_order}_first")
if args.succint:
	output_dir = output_dir + "_succint"

def get_model(name):
	if name == "gpt":
		model = OpenAIWrapper("text-davinci-003")
	elif name == "gpt_turbo":
		model = OpenAIWrapper("gpt-3.5-turbo-instruct")
	elif name == "ft":
		model = FT_model()
	elif name == "llama":
		model = Probe_model("decapoda-research/llama-7b-hf")
	else:
		raise NotImplementedError
	return model


head_pattern = "rule{rule_index}_value_cache.csv_head"
tail_pattern = "rule{rule_index}_value_cache.csv_longtail"
model = get_model(args.model)
for rule_index in rule_indexes:
	logger.info("Get llh for rule {}".format(rule_index))

	path_d = {"head":os.path.join(path,head_pattern.format(rule_index = rule_index)),"tail":os.path.join(path,tail_pattern.format(rule_index = rule_index))}
	try:
		handler = Handler(path_d,args,output_dir,args.plot_as_pre_con,args.traverse_order_entity_force)
	except MyError as e:
		logger.error("We receive path like {}, but we dont have such rule".format(path_d))
		continue

	_datas = handler.partial_rule_map
	_datas_word = handler.partial_word_map
	datas = ddict(dict)
	datas["tail"]["rules"] = _datas["tail"]
	datas["head"]["rules"] = _datas["head"]
	datas["tail"]["words"] = _datas_word["tail"]
	datas["head"]["words"] = _datas_word["head"]

	try:
		with open(f"{handler.output_dir}/max_node_len.json") as f:
			d = json.load(f)
	except:
		d = {}

	d.update({f"{handler._which_rule}":handler.max_node_for_rule})
	logger.info(f"Update max_node_len by {handler._which_rule} : {handler.max_node_for_rule}")

	with open(f"{handler.output_dir}/max_node_len.json","w") as f:
		json.dump(d,f)


	if os.path.exists(f"{handler.output_dir}/{handler._which_rule}_agreement_{args.model}.json"):
		logger.info(f"{args.model} already has llh file for {handler._which_rule}")
		continue

	all_scores = ddict(dict)
	for d in [score(args.model,model,"tail",datas),score(args.model,model,"head",datas)]:
		for key in d:
			all_scores[key][list(d[key].keys())[0]] = d[key][list(d[key].keys())[0]]

	with open(f"{handler.output_dir}/{handler._which_rule}_agreement_{args.model}.json","w") as f:
		json.dump(all_scores,f)
	logger.info(f"Save llf of {handler._which_rule} from {args.model} ")




















	

