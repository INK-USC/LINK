import os
from itertools import combinations
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM,LlamaTokenizer,LlamaForCausalLM
import openai
from collections import defaultdict
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
import backoff
import csv
import random
random.seed(42)
from copy import deepcopy
import re
import json
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch.nn.functional as F
from numpy.linalg import norm
from numpy import dot
import tiktoken
import pandas as pd
import math
from collections import defaultdict as ddict
import jsonlines
import pathlib
import evaluate
rouge = evaluate.load("rouge")
os.environ["TIKTOKEN_CACHE_DIR"] = ""
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY") 


def batch_process(l,batch_size):
    for i in range(0,len(l),batch_size):
        yield l[i:i+batch_size]


def get_indices_of_largest_values(lst,needed = 1):
    if len(lst) == 0:
        return []
    if isinstance(lst[0],str):
        return []
    enumerated_list = list(enumerate(lst))
    sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=True)
    indices = [index for index, _ in sorted_list[:needed]]
    
    return indices


def _get_full_rule_predicates(traverse_order,rule,graph, traverse_order_entity):
    searched = []
    searched.append(traverse_order[0])
    dt = rule.variables[traverse_order[0]].datatype
    searched_rules = []
    for traverse_order_index in range(1, len(traverse_order)):
        rules_to_search = []
        for node in searched:
            if node in graph[traverse_order[traverse_order_index]]:
                r = graph[traverse_order[traverse_order_index]][node]
                rules_to_search.append(r)
                if r not in searched_rules:
                    searched_rules.append(r)
        if traverse_order_index != 1:
            # remove conclusion rule when premise first
            rules_to_search = list(filter(lambda r: r not in rule.conclusion, rules_to_search))
        var = traverse_order[traverse_order_index]
        searched.append(var)

        if traverse_order_entity == "premise":
            try:
                tobe_swaped_index = searched_rules.index(list(rule.conclusion_map.keys())[0])
                searched_rules[tobe_swaped_index], searched_rules[-1] = searched_rules[-1], searched_rules[tobe_swaped_index]
            except:
                pass

    return searched_rules


def _swap_conclusion_to_last(predicates,conclusion_predicates):
    
    try:
        tobe_swaped_index = predicates.index(conclusion_predicates)
        predicates[tobe_swaped_index], predicates[-1] = predicates[-1], predicates[tobe_swaped_index]
    except:
        pass
    return predicates


@dataclass
class Variable:
    name: str
    datatype: str
    is_generic: bool
    nodetype: str = None # choices: "factual"


class Rule:
    def __init__(self, premise_map, conclusion_map, variables):
        self.premise = list(premise_map.keys())
        self.premise_map = premise_map
        self.conclusion = list(conclusion_map.keys())
        self.conclusion_map = conclusion_map # assume conclusion is a single predicate
        self.variables = variables

    def get_all_variable_names(self):
        return set([variable.name for variable in self.variables.values()])

    def __str__(self):
        string = f"Premise: {self.premise}\nConclusion: {self.conclusion}\nVariables: {self.variables}"
        return string

    def parse_rule(self):
        """
        example_rule = "allergic_to(Person X, Allergen A) & ingredient_in(Ingredient Z, Dish B) & one_type_of(Ingredient Z, Allergen A) ⇒ cannot_eat(Person X, Dish B)"
        # if a component contains generics, they are information edges, otherwise they are constraint edges
        # if a node appears in information edges, they are information nodes, otherwise they are constraint nodes
        
       example_output = {'root': {'X': None}, 
                        'X': {'A': 'allergic_to(X, A)', 'B': 'cannot_eat(X, B)'}, 
                        'A': {'X': 'allergic_to(X, A)', 'Z': 'one_type_of(Z, A)'},
                        'Z': {'A': 'one_type_of(Z, A)', 'B': 'ingredient_in(Z, B)'}, 
                        'B': {'X': 'cannot_eat(X, B)', 'Z': 'ingredient_in(Z, B)'}
                        },
        """

        nodes = defaultdict(dict)
        information_edges = set()
        constraint_edges = set()
        conclusion_edges = set()

        conclusion_first_order = []

        # get conclusion nodes
        for predicate in self.conclusion:
            predicate_nodes =  [p.strip() for p in predicate[predicate.index("(")+1: predicate.index(")")].split(",")]
            conclusion_edges.add(predicate)

            for c_n in predicate_nodes:
                c_n = c_n.strip()
                if self.variables[c_n].is_generic:
                    nodes["root"][c_n] = None
                    # first item is generic node
                    conclusion_first_order.append(c_n)
            for c_n in predicate_nodes:
                c_n = c_n.strip()
                if not self.variables[c_n].is_generic:
                    # then add non-generic nodes in conclusion
                    conclusion_first_order.append(c_n)

                    for g_n in nodes["root"]:
                        nodes[g_n][c_n] = predicate
                        nodes[c_n][g_n] = predicate
                        self.variables[c_n].nodetype = "information"

        # first add information edges and information nodes
        for predicate in self.premise:
            # component = live_in(Tom, A)
            predicate_nodes = [p.strip() for p in predicate[predicate.index("(")+1: predicate.index(")")].split(",")]
            for g_n in nodes["root"]:
                if g_n in predicate_nodes:
                    for c_n in predicate_nodes:
                        if c_n != g_n:
                            nodes[g_n][c_n] = predicate
                            nodes[c_n][g_n] = predicate
                            self.variables[c_n].nodetype = "information"
                            information_edges.add(predicate)
        # then add constraint edges
        for predicate in self.premise:
            predicate_nodes = [p.strip() for p in predicate[predicate.index("(")+1: predicate.index(")")].split(",")]
            
            for n1 in predicate_nodes[:-1]:
                for n2 in predicate_nodes[1:]:
                    nodes[n1][n2] = predicate
                    nodes[n2][n1] = predicate
                    if self.variables[n1].nodetype is None:
                        self.variables[n1].nodetype = "constraint"

                    if self.variables[n2].nodetype is None: 
                        self.variables[n2].nodetype = "constraint"

        constraint_edges = set(self.premise) - information_edges

        # get search order of premise nodes
        current_node = conclusion_first_order[-1]
        covered_constraint_edges = set()
        
        while len(covered_constraint_edges) < len(constraint_edges):

            for predicate in constraint_edges:
                # skip predicates that have already covered nodes
                if predicate in covered_constraint_edges:
                    continue
                predicate_nodes =  [p.strip() for p in predicate[predicate.index("(")+1: predicate.index(")")].split(",")]
                # add the next node to order from predicate that contains current node
                if current_node == predicate_nodes[0].strip():
                    c_n = predicate_nodes[1].strip()
                    if c_n not in conclusion_first_order:
                        conclusion_first_order.append(c_n)
                        covered_constraint_edges.add(predicate)
                        current_node = c_n
                        break
                elif current_node == predicate_nodes[1].strip():
                    c_n = predicate_nodes[0].strip()
                    if c_n not in conclusion_first_order:
                        conclusion_first_order.append(c_n)
                        covered_constraint_edges.add(predicate)
                        current_node = c_n
                        break
                
        for node in self.variables:
            if node not in conclusion_first_order:
                conclusion_first_order.append(node)
        to_be_reversed = conclusion_first_order[1:]
        to_be_reversed.reverse()
        premise_first_order = conclusion_first_order[:1] + to_be_reversed

        return nodes, information_edges, constraint_edges, conclusion_first_order, premise_first_order

    def verbalize_with_partial_rules(self, predicates, values, nodes):
        """
            given predicates that are previously searched, verbalize by substituting the nodes with values
        """
        output = []
        for p in predicates:
            items = p[p.index("(")+1: p.index(")")].split(",")
            items = [i.strip() for i in items]
            keep = True
            for item in items:
                if item not in nodes:
                    keep = False
            if not keep:
                continue
            if p in self.premise_map:
                nl = self.premise_map[p]
            elif p in self.conclusion_map:
                nl = self.conclusion_map[p]
            else:
                raise ValueError(f"predicate {p} not found in rule")
            for n in nodes:
                if n.strip() in values:
                    nl = nl.replace(f"[{n.strip()}]", values[n.strip()])
            output.append(nl)
        return " and ".join(output)

    def convert_search_result_to_questions(self, predicates, values, nodes, var):
        """
            Creates prompt questions for verifier.
        """
        pos = "yes"
        neg = "no"
        additional_prompt = "For the following question, please answer in a normal life scenario with no special consideration:\n"
        output = []
        for p in predicates:
            nodes = p[p.index("(")+1: p.index(")")].split(",")
            nodes = [n.strip() for n in nodes]
            # only include predicates that contain the variable and do not contain generics
            if var not in nodes:
                continue
            has_generic = False
            for node in nodes:
                node = node.strip()
                if self.variables[node].is_generic:
                    has_generic = True
                    break
            if has_generic:
                continue
            if p in self.premise_map:
                nl = self.premise_map[p]
            elif p in self.conclusion_map:
                nl = self.conclusion_map[p]
            else:
                raise ValueError(f"predicate {p} not found in rule")
            for n in nodes:
                if n.strip() in values:
                    v  = " ".join(values[n.strip()].split("_"))
                    nl = nl.replace(f"[{n.strip()}]", v)
            # Prompt to verify factual correctness
            prompt = additional_prompt + f"Question: Is the following statement correct? Please answer {pos} or {neg}\nStatement: {nl}\nAnswer:"
            output.append(prompt)
        # Prompt to verify datatype correctness
        datatype = self.variables[var].datatype
        datatype_prompt = additional_prompt + f"Question: Is the following statement correct? Please answer {pos} or {neg}\n{' '.join(values[var].strip().split('_'))} is a {datatype.lower()}\nAnswer:"
        return output, datatype_prompt


class BeamSearchGenerator:
    def __init__(self, knowledge_model, likelihood_model, verifier_model, rule, graph, traverse_order, beam_size, traverse_order_entity, rank_as_pre_con):
        self.beam_size = beam_size
        self.rule = rule
        self.graph = graph
        self.traverse_order = traverse_order
        self.beams = {"head": {}, "tail": {}}
        # {(root, p): [{root: v1, p: v2}, ...]...}
        self.searched = []
        self.knowledge_model = knowledge_model
        self.likelihood_model = likelihood_model
        self.verifier_model = verifier_model
        self.traverse_order_entity = traverse_order_entity
        self.rank_as_pre_con = rank_as_pre_con
    
    def search(self, n_sample, use_ft = False, get_verifier_samples = False, deduplicate=False, bias_value=-100, factual_verifier_threshold=0.2, datatype_verifier_threshold=0.2, dynamic_verifier_threshold=False, factual_lowest_verifier_threshold=0.75, datatype_lowest_verifier_threshold=0.75, dynamic_threshold_step=0.05, accumulate_verifier_confidence=False, weighted_accumulate=False, use_nl_prompt=False, dynamic_ranker=False, ranker_accept_ratio=0.75):
        """
        knowledge beam search
        """
        self.searched.append(self.traverse_order[0])
        dt = self.rule.variables[self.traverse_order[0]].datatype
        self.beams["head"][tuple(self.searched)] = [{self.traverse_order[0]: f"{dt}X"}]
        self.beams["tail"][tuple(self.searched)] = [{self.traverse_order[0]: f"{dt}X"}]

        factual_verifier_samples = []
        datatype_verifier_samples = []
        pbar = tqdm(total = len(self.traverse_order),desc="search and rank by ll")
        searched_rules = []
        saved_beam_values = ddict(list)
        verify_output_cache = ddict(dict)

        for traverse_order_index in range(1, len(self.traverse_order)):
            rules_to_search = []
            for node in self.searched:
                if node in self.graph[self.traverse_order[traverse_order_index]]:
                    r = self.graph[self.traverse_order[traverse_order_index]][node]
                    rules_to_search.append(r)
                    if r not in searched_rules:
                        searched_rules.append(r)
            if traverse_order_index != 1:
                # remove conclusion rule when premise first
                rules_to_search = list(filter(lambda r: r not in self.rule.conclusion, rules_to_search))
            

            if self.traverse_order_entity == "premise":
                try:
                    tobe_swaped_index = searched_rules.index(list(self.rule.conclusion_map.keys())[0])
                    searched_rules[tobe_swaped_index], searched_rules[-1] = searched_rules[-1], searched_rules[tobe_swaped_index]
                except:
                    pass


            fout.write(str(searched_rules))
            fout.write(nl)


            var = self.traverse_order[traverse_order_index]
            datatype = self.rule.variables[var].datatype
            nodetype = self.rule.variables[var].nodetype
            for distribution in ["head", "tail"]:

                if traverse_order_index == 1:
                    _n_sample = n_sample * 2
                else:
                    _n_sample = n_sample
                if traverse_order_index == 1 and distribution == "tail":
                    # since rank_index is based on head, so we wanna last self.beam_size
                    # do not filter out by ranker in the first step
                    beam_selected_index = rank_index
                    beam_values = [_beam_values[_] for _ in beam_selected_index]
                    self.beams[distribution][tuple(self.searched + [var])] = beam_values # Note: here we only store values not the verbalized predicates
                    continue

                previous_beams = self.beams[distribution][tuple(self.searched)]
                beam_values = []
                correct_beam_values = []
                original_beam_indexs = []
                _saved_beam_values = []

                for values in previous_beams:
                    prompt = self.knowledge_model.construct_prompt(var, rules_to_search, values, datatype=datatype, nodetype=nodetype, beam_size=_n_sample, premise_map=self.rule.premise_map, conclusion_map=self.rule.conclusion_map, nl_prompt=use_nl_prompt)

                    if nodetype == "factual":
                        
                        def _search_one_factual_value(prompt, var, datatype):
                            # search 1 value for factual rule
                            output = self.knowledge_model.generate([prompt], temperature=0.7, max_tokens=512)[0][0].strip()
                            # extract variable values from output
                            # post-processing will examine whether the output follows certain format
                            variable_values = self.knowledge_model.post_process(output, var, datatype)
                            # search_times
                            search_times = 1
                            # making sure there are valid values (limit max search_times)
                            while len(variable_values) == 0 and search_times <= 10:
                                output = self.knowledge_model.generate([prompt], temperature=0.7, max_tokens=512)[0][0].strip()
                                variable_values = self.knowledge_model.post_process(output, var, datatype)
                                search_times += 1
                            return variable_values
                        # first search
                        variable_values = _search_one_factual_value(prompt, var, datatype)
                        if len(variable_values) != 0:
                            # if we can get one factual value, prompt again to verify
                            variable_values_2 = _search_one_factual_value(prompt, var, datatype)
                        else:
                            variable_values_2 = []
                        # verify for factual values
                        correct_beam_values, original_beam_indexs, _saved_beam_values, beam_values, verify_output_cache = self._verify_factual_node(beam_values, variable_values, variable_values_2, values,
                                                                                                                                                    distribution, traverse_order_index, var,
                                                                                                                                                    correct_beam_values, original_beam_indexs, _saved_beam_values,
                                                                                                                                                    verify_output_cache)
                    else:
                        # search {call_times} times, {n_sample} values once -> {beam_size} values in total
                        assert self.beam_size % n_sample == 0, "beam_size should be the multiple of n_sample"
                        call_times = int(self.beam_size / n_sample)
                        factual_dynamic_start_threshold = factual_verifier_threshold
                        previous_correct_beam_values_index = len(correct_beam_values)
                        previous_original_beam_index = len(original_beam_indexs)

                        if deduplicate and call_times != 1:
                            # deduplicate with logit_bias
                            logit_bias = {}
                            outputs = []
                            all_correct_values = []
                            base_prompt = prompt
                            break_flags = []
                            # set the threshold to 0 in the beginning, then it would must be updated in the first call
                            factual_dynamic_start_threshold = 0
                            factual_start_threshold_change_flag = False
                            for i in range(call_times):
                                output = self.knowledge_model.generate([prompt], temperature=0.7, max_tokens=512, logit_bias=logit_bias)[0][0].strip()
                                search_times = 1
                                while _n_sample != 1 and len(output.split("\n")) == 1 and search_times <= 10:
                                    # make sure that each value is separated by \n (limit max search_times)
                                    output = self.knowledge_model.generate([prompt], temperature=0.7, max_tokens=512, logit_bias=logit_bias)[0][0].strip()
                                    search_times += 1
                                # filter out values longer than 64
                                split_output = list(filter(lambda o: len(o) <= 64, output.split("\n")))
                                output = '\n'.join(split_output)
                                # post-process searched values
                                tmp_values = set(self.knowledge_model.post_process(output.strip()))
                                # print("tmp_values: {}".format(tmp_values))
                                search_times = 1
                                while (len(tmp_values) == 0 and search_times <= 10) or (len(list(tmp_values - set(all_correct_values)))== 0 and search_times <= 10):
                                    # make sure there are valid values (limit max search_times)
                                    output = self.knowledge_model.generate([prompt], temperature=0.7, max_tokens=512, logit_bias=logit_bias)[0][0].strip()
                                    split_output = list(filter(lambda o: len(o) <= 64, output.split("\n")))
                                    output = '\n'.join(split_output)
                                    tmp_values = set(self.knowledge_model.post_process(output.strip()))
                                    search_times += 1
                                outputs.append(output)
                                # remove duplicates
                                tmp_values = list(tmp_values - set(all_correct_values))
                                if len(tmp_values) == 0:
                                    logger.info("We can not get more valid options")
                                    break
                                   
                                logger.info("tmp values: {}".format(tmp_values))
                                correct_beam_values, original_beam_indexs, _saved_beam_values, beam_values, correct_values, wrong_values, verify_output_cache, accepted_num, factual_dynamic_start_threshold, change_flag = self._verify_after_call(beam_values, tmp_values, values,
                                                                                                                                                                                                                                                        distribution, traverse_order_index, rules_to_search, var, get_verifier_samples,
                                                                                                                                                                                                                                                        datatype_verifier_samples, factual_verifier_samples, datatype_verifier_threshold, factual_verifier_threshold,
                                                                                                                                                                                                                                                        dynamic_verifier_threshold, datatype_lowest_verifier_threshold, factual_lowest_verifier_threshold, dynamic_threshold_step,
                                                                                                                                                                                                                                                        factual_dynamic_start_threshold,
                                                                                                                                                                                                                                                        correct_beam_values, original_beam_indexs, _saved_beam_values,
                                                                                                                                                                                                                                                        verify_output_cache, accumulate_verifier_confidence, weighted_accumulate)
                                if change_flag and i != 0:
                                    # the start threshold has been updated in the later calls 
                                    factual_start_threshold_change_flag = True
                                logger.info("accepted values: {}".format(correct_values))
                                logger.info("rejected values: {}".format(wrong_values))
                                if accepted_num[0] == 0 or accepted_num[1] == 0:
                                    # No value passes the verifier
                                    break_flag = True
                                else:
                                    break_flag = False
                                break_flags.append(break_flag)
                                if len(break_flags) >= 2 and break_flags[-1] and break_flags[-2]:
                                    # if the result of two consecutive searches are all rejected by verifier, bail the beam
                                    logger.info("Bail early because consecutive searches all have low verifier probability.")
                                    logger.info(f"break_flags: {break_flags}")
                                    break
                                # forbid genearting existed values in prompt
                                all_correct_values.extend(correct_values)
                                if len(all_correct_values) > 0:
                                    forbid_correct_values = ", ".join(all_correct_values)
                                    prompt = base_prompt + f" Do not generate these values: {forbid_correct_values}."

                                # add logit_bias when the limit is not reached
                                if len(logit_bias) < 300:
                                    # only add tokens of rejected values
                                    forbid_token_str = []
                                    for value in wrong_values:
                                        if not bool(re.search(r'\d', value)):
                                            # only add values without numbers
                                            forbid_token_str.append(f" {value}")
                                    forbid_token_str = "\n".join(forbid_token_str)
                                    tokens = set(self.knowledge_model.tiktoken.encode(forbid_token_str))
                                    # do not change the likelihood of tokens in template
                                    for template_token in ["\n", ".", " ."]:
                                        tokens.discard(self.knowledge_model.tiktoken.encode(template_token)[0])
                                    for num_token in range(1, _n_sample+1):
                                        tokens.discard(self.knowledge_model.tiktoken.encode(str(num_token))[0])
                                    for token in tokens:
                                        if len(logit_bias) < 300:
                                            logit_bias[token] = bias_value
                                        else:
                                            # reach the limit
                                            break

                            # recheck all accepted values
                            if factual_start_threshold_change_flag:
                                # correct_beam_values, original_beam_indexs, _save_beam_values
                                new_correct_beam_values = []
                                new_original_beam_indexs = []
                                for correct_beam, original_index in zip(correct_beam_values[previous_correct_beam_values_index:], original_beam_indexs[previous_original_beam_index:]):
                                    factual_ver = _saved_beam_values[original_index]["factual_ver"]
                                    key = "accumulated_output" if accumulate_verifier_confidence else "verifier_output"
                                    _saved_beam_values[original_index]["verifier_threshold"] = factual_dynamic_start_threshold
                                    if factual_ver[key][0] >= factual_dynamic_start_threshold:
                                        # only accept whose confidence > final threshold
                                        new_correct_beam_values.append(correct_beam)
                                        new_original_beam_indexs.append(original_index)
                                    else:
                                        _saved_beam_values[original_index]["stamp"] += "-out_verifier_factual"
                                # update
                                previous_correct_beam_values = correct_beam_values[:previous_correct_beam_values_index]
                                previous_original_beam_indexs = original_beam_indexs[:previous_original_beam_index]
                                correct_beam_values = previous_correct_beam_values + new_correct_beam_values
                                original_beam_indexs = previous_original_beam_indexs + new_original_beam_indexs
                        else:
                            outputs = self.knowledge_model.generate([prompt], num_sample=call_times, temperature=0.7, max_tokens=512)[0]
                            # post-process searched values
                            variable_values = []
                            for o in outputs:
                                variable_values += self.knowledge_model.post_process(o.strip())
                            # deduplicate
                            variable_values = list(set(variable_values))
                            # print(f'Get {len(variable_values)} values: {variable_values}\n')

                            correct_beam_values, original_beam_indexs, _saved_beam_values, beam_values, _, _, verify_output_cache, _, _, _ = self._verify_after_call(beam_values, variable_values, values,
                                                                                                                                                            distribution, traverse_order_index, rules_to_search, var, get_verifier_samples,
                                                                                                                                                            datatype_verifier_samples, factual_verifier_samples, datatype_verifier_threshold, factual_verifier_threshold,
                                                                                                                                                            dynamic_verifier_threshold, datatype_lowest_verifier_threshold, factual_lowest_verifier_threshold, dynamic_threshold_step,
                                                                                                                                                            factual_dynamic_start_threshold,
                                                                                                                                                            correct_beam_values, original_beam_indexs, _saved_beam_values,
                                                                                                                                                            verify_output_cache, accumulate_verifier_confidence, weighted_accumulate)

                logger.info(f"beam_values:{beam_values}")

                beam_values = correct_beam_values

                _beam_values = beam_values # this is for caching the first variable beam only

                if (use_ft and traverse_order_index <= 2) or not use_ft:
                    # Use likelihood model keep top-k beam size of most "likely" values and least "likely" values in beam_values
                    # Note that this is only ranking premises concerning the current variables, which is different from "longtailness" of the entire sentence

                    # annotation: l -> list, ll -> likelihood
                    partial_sequence_l = []
                    logger.info("rules_to_search_for_ranking: {}".format(searched_rules))

                    partial_sequence_l = [self._create_partial_sequence_for_ranking(predicates=searched_rules, values=value_dict, nodes=self.searched + [var]) for value_dict in beam_values]
                    logger.info("stmts for ranking: {}".format(partial_sequence_l))

                    partial_sequence_l_ll = self.likelihood_model.calculate_likelihood(partial_sequence_l,batch_size = args.batch_size)


                    # below is a experimental part
                    # ************************
                    # Hypothese: we find that in rule 4, food related rules, the head rule always contain high word overlapping which we wanna avoid
                    if distribution == "head":
                        def for_repetition_scores(d,traverse_order_index,all_traverse_order):
                            references= []
                            predictions = []
                            for i in range(traverse_order_index):
                                
                                references.append(d[all_traverse_order[i]])
                                predictions.append(d[all_traverse_order[traverse_order_index]])
                            return dict(references=references,predictions=predictions)

                        for_repetition_scores_l = [for_repetition_scores(value_dict,traverse_order_index,self.traverse_order) for value_dict in beam_values]
                        for_repetition_scores_l = [rouge.compute(predictions = _["predictions"], references= _["references"])["rougeL"] for _ in for_repetition_scores_l]
                        
                        partial_sequence_l_ll = list(np.array(partial_sequence_l_ll) - np.array(for_repetition_scores_l) * args.repetition_penalty)
                    # ************************


                    # For head, we want the high logprobs reverse = True
                    # For longtail, we want the low logprobs. reverse = False
                    rank_index = sorted(list(range(len(partial_sequence_l_ll))),key = lambda x: partial_sequence_l_ll[x],reverse= True if distribution == "head" else False)

                elif traverse_order_index > 2 and use_ft:

                    beam_entities = create_entities_pairs(beam_values,self.traverse_order[0])
                    partial_entities_l_ll = self.likelihood_model.calculate_likelihood(beam_entities,fasttext=True,batch_size = args.batch_size)
                    rank_index = sorted(list(range(len(partial_entities_l_ll))),key = lambda x: partial_entities_l_ll[x],reverse= True if distribution == "head" else False)

                if dynamic_ranker:
                    ranker_accept_size = min(self.beam_size, math.ceil(ranker_accept_ratio * len(beam_values)))
                else:
                    ranker_accept_size = self.beam_size
                if traverse_order_index == 1:
                    # do not filter out by ranker in the first step
                    beam_selected_index = rank_index
                else:
                    beam_selected_index = rank_index[:ranker_accept_size]
                beam_values = [beam_values[_] for _ in beam_selected_index]


                choosen_by_ranker_index = [original_beam_indexs[_] for _ in beam_selected_index]
                filter_by_ranker_index = list(set(original_beam_indexs) - set(choosen_by_ranker_index))
                for i in filter_by_ranker_index:
                    _saved_beam_values[i]["stamp"] += "-out_ranker"



                self.beams[distribution][tuple(self.searched + [var])] = beam_values # Note: here we only store values not the verbalized predicates
                
                # record all beams and categorize them into out_verifier or out_ranker, which means the reason why the beam is filtered
                if traverse_order_index == 1:
                    for _ in _saved_beam_values:
                        # do not seperate head/tail in the first step
                        saved_beam_values["head"].append(_)
                        longtail_ = deepcopy(_)
                        longtail_["stamp"] = longtail_["stamp"].replace("head-","tail-")
                        saved_beam_values["tail"].append(longtail_)
                else:
                    saved_beam_values[f"{distribution}"].extend(_saved_beam_values)

            self.searched.append(var)
            pbar.update(1)
        return self.beams["head"][tuple(self.searched)], self.beams["tail"][tuple(self.searched)], factual_verifier_samples, datatype_verifier_samples, saved_beam_values, verify_output_cache

    def _create_partial_sequence_for_ranking(self, predicates, values, nodes):
        """
        create partial prompt for likelihood model
        """
        # predicates are the predicates entirely in conclusion or premise
        _predicates = deepcopy(predicates)

        if self.rank_as_pre_con:
            _predicates = _swap_conclusion_to_last(_predicates,list(self.rule.conclusion_map.keys())[0])
        partial_sequence = self.rule.verbalize_with_partial_rules(_predicates, values, nodes)
        return partial_sequence

    def _create_partial_sequence_for_verifying(self, predicates, values, nodes, var):
        """
        create partial prompt for verifier model
        """
        # predicates are only the predicates concerning the current variable
        factual_sequences, datatype_sequence = self.rule.convert_search_result_to_questions(predicates, values, nodes, var)
        return factual_sequences, datatype_sequence

    def _get_previous_confidence_for_verifying(self, verify_output_cache, values, searched_nodes):
        """
        get the previous confidence of searched values from confidence cache
        """
        searched_values = {}
        for node in searched_nodes:
            searched_values[node] = values[node]
        if verify_output_cache is None:
            return ([], []), ([], [])
        elif len(searched_values) == 1:
            # only generic node
            return ([], []), ([], [])
        else:
            previous_key = "_".join(searched_values.values())
            previous_confidence_datatype = verify_output_cache[previous_key]["datatype"] if "datatype" in verify_output_cache[previous_key] else ([], [])
            previous_confidence_factual = verify_output_cache[previous_key]["factual"] if "factual" in verify_output_cache[previous_key] else ([], [])
            return previous_confidence_factual, previous_confidence_datatype

    def _verify(self, rules_to_search, beam_values, var, traverse_order_index, get_verifier_samples,
                datatype_verifier_samples, factual_verifier_samples, _saved_beam_values,
                datatype_verifier_threshold, factual_verifier_threshold,
                dynamic_verifier_threshold, datatype_lowest_verifier_threshold, factual_lowest_verifier_threshold, dynamic_threshold_step,
                factual_dynamic_start_threshold,
                verify_output_cache, accumulate_verifier_confidence, weighted_accumulate):
        """
        call verfier model to filter the search results with dynamic threshold
        """
        # l -> a list, ll -> a list of lists
        # separate factual and datatype questions
        factual_sequence_ll, datatype_question_l = zip(*[self._create_partial_sequence_for_verifying(predicates=rules_to_search, values=value_dict, nodes=self.searched + [var], var=var) for value_dict in beam_values])
        datatype_question_ll = [[_] for _ in datatype_question_l]
        logger.info("factual_sequence_ll: {}".format(factual_sequence_ll))
        logger.info("datatype_question_ll: {}".format(datatype_question_ll))
        factual_pre_confidence, datatype_pre_confidence = zip(*[self._get_previous_confidence_for_verifying(verify_output_cache, values=value_dict, searched_nodes=self.searched) for value_dict in beam_values])
        logger.info("factual_pre_confidence: {}".format(factual_pre_confidence))
        logger.info("datatype_pre_confidence: {}".format(datatype_pre_confidence))
        datatype_result_l, samples_datatype = self.verifier_model.verify(datatype_question_ll, get_verifier_samples=get_verifier_samples, verifier_threshold=datatype_verifier_threshold, batch_size = args.batch_size, previous_confidence=datatype_pre_confidence, accumulate_verifier_confidence=accumulate_verifier_confidence, weighted_accumulate=weighted_accumulate)
        # dynamic threshold for datatype verifier
        def check_threshold(samples, threshold, accumulate_verifier_confidence):
            key = "accumulated_output" if accumulate_verifier_confidence else "verifier_output"
            accept_l = list(filter(lambda o: o[key][0] >= threshold, samples))
            return len(accept_l)
        datatype_dynamic_verifier_threshold = datatype_verifier_threshold
        datatype_accept_num = check_threshold(samples_datatype, datatype_verifier_threshold, accumulate_verifier_confidence)
        if dynamic_verifier_threshold:
            while datatype_accept_num == 0 and datatype_dynamic_verifier_threshold > datatype_lowest_verifier_threshold:
                # decrease if no accepted values & higher than mininum threshold
                old_datatype_dynamic_verifier_threshold = datatype_dynamic_verifier_threshold
                datatype_dynamic_verifier_threshold -= dynamic_threshold_step
                logger.info(f"Lower datatype threshold: {old_datatype_dynamic_verifier_threshold} -> {datatype_dynamic_verifier_threshold}")
                datatype_accept_num = check_threshold(samples_datatype, datatype_dynamic_verifier_threshold, accumulate_verifier_confidence)
            # update results
            for i in range(len(samples_datatype)):
                key = "accumulated_output" if accumulate_verifier_confidence else "verifier_output"
                if samples_datatype[i][key][0] >= datatype_dynamic_verifier_threshold:
                    datatype_result_l[i] = self.verifier_model.positive_token
                else:
                    datatype_result_l[i] = self.verifier_model.negative_token
                samples_datatype[i]["verifier_threshold"] = datatype_dynamic_verifier_threshold
        if get_verifier_samples:
            logger.info("samples_datatype: {}".format(samples_datatype))
            datatype_verifier_samples.extend(samples_datatype)
            for i in range(len(samples_datatype)):
                value_str = "_".join(beam_values[i].values())
                verify_output_cache[value_str]["datatype"] = samples_datatype[i]["accumulated_output_list"]
        factual_result_l, samples_factual = self.verifier_model.verify(factual_sequence_ll, get_verifier_samples=get_verifier_samples, verifier_threshold=factual_dynamic_start_threshold, batch_size = args.batch_size, previous_confidence=factual_pre_confidence, accumulate_verifier_confidence=accumulate_verifier_confidence, weighted_accumulate=weighted_accumulate)
        # check the max confidence in this call
        def get_highest_confidence_threshold(samples, accumulate_verifier_confidence, dynamic_threshold_step):
            key = "accumulated_output" if accumulate_verifier_confidence else "verifier_output"
            confidences = [o[key][0] for o in samples]
            if len(confidences):
                highest_confidence = max(confidences)
            else:
                highest_confidence = 0
            return (highest_confidence // dynamic_threshold_step) * dynamic_threshold_step
        highest_confidence_threshold = get_highest_confidence_threshold(samples_factual, accumulate_verifier_confidence, dynamic_threshold_step)
        if highest_confidence_threshold > factual_dynamic_start_threshold:
            # higher confidence, change starting threshold
            # the threshould cannot exeed the highest threshold (0.85)
            factual_dynamic_start_threshold = min(highest_confidence_threshold, factual_verifier_threshold)
            change_flag = True
        else:
            change_flag = False
        factual_accept_num = check_threshold(samples_factual, factual_dynamic_start_threshold, accumulate_verifier_confidence)
        for i in range(len(samples_factual)):
            key = "accumulated_output" if accumulate_verifier_confidence else "verifier_output"
            if samples_factual[i][key][0] >= factual_dynamic_start_threshold:
                factual_result_l[i] = self.verifier_model.positive_token
            else:
                factual_result_l[i] = self.verifier_model.negative_token
            samples_factual[i]["verifier_threshold"] = factual_dynamic_start_threshold
        if get_verifier_samples:
            logger.info("samples_factual: {}".format(samples_factual))
            factual_verifier_samples.extend(samples_factual)
            for i in range(len(samples_factual)):
                value_str = "_".join(beam_values[i].values())
                verify_output_cache[value_str]["factual"] = samples_factual[i]["accumulated_output_list"]
    
        for i in range(len(samples_datatype)):
            _saved_beam_values[i]["datatype_ver"] = samples_datatype[i]
            try:
                _saved_beam_values[i]["factual_ver"] = samples_factual[i]
            except:
                _saved_beam_values[i]["factual_ver"] = None

        # count the number of accepted values
        accepted_num = [factual_accept_num, datatype_accept_num]


        def filter_by_factual_datatype(datatype_result_l,samples_datatype,factual_result_l,samples_factual, which_verifier_first):
            if which_verifier_first == "factual":
                first_result_l, samples_first = factual_result_l,samples_factual
                second_result_l, samples_second = datatype_result_l,samples_datatype
            else:
                first_result_l, samples_first = datatype_result_l,samples_datatype
                second_result_l, samples_second = factual_result_l,samples_factual


            must_save_at_first_search = False
            if len(samples_first) == 0:
                must_save_at_first_search = True

            temp_beam_values = []
            temp_beam_indexes = []
            correct_beam_values = []
            original_beam_indexs = []
            for i in range(len(beam_values)):
                if first_result_l[i] == self.verifier_model.positive_token or must_save_at_first_search:
                    temp_beam_values.append(beam_values[i])
                    temp_beam_indexes.append(i)
                else:
                    if which_verifier_first == "datatype":
                        _saved_beam_values[i]["stamp"] += "-out_verifier_datatype"
                    else:
                        _saved_beam_values[i]["stamp"] += "-out_verifier_factual"


        
            must_save_at_second_search = False
            if len(samples_second) == 0:
                must_save_at_second_search = True

            for i in temp_beam_indexes:
                if second_result_l[i] == self.verifier_model.positive_token or must_save_at_second_search:
                    correct_beam_values.append(beam_values[i])
                    original_beam_indexs.append(i)
                else:
                    if which_verifier_first == "datatype":
                        _saved_beam_values[i]["stamp"] += "-out_verifier_factual"
                    else:
                        _saved_beam_values[i]["stamp"] += "-out_verifier_datatype"


            return correct_beam_values, original_beam_indexs, _saved_beam_values

        return filter_by_factual_datatype(datatype_result_l,samples_datatype, factual_result_l, samples_factual, args.which_verifier_first), verify_output_cache, accepted_num, factual_dynamic_start_threshold, change_flag

    def _verify_after_call(self, beam_values, variable_values, values,
                           distribution, traverse_order_index, rules_to_search, var, get_verifier_samples,
                           datatype_verifier_samples, factual_verifier_samples, datatype_verifier_threshold, factual_verifier_threshold,
                           dynamic_verifier_threshold, datatype_lowest_verifier_threshold, factual_lowest_verifier_threshold, dynamic_threshold_step,
                           factual_dynamic_start_threshold,
                           correct_beam_values, original_beam_indexs, _saved_beam_values,
                           verify_output_cache, accumulate_verifier_confidence, weighted_accumulate
                           ):
        """
        verify search results after each search
        """
        tmp_beam_values = []
        previous_idx = len(beam_values)
        for v in variable_values:
            new_values = deepcopy(values)
            new_values[var] = v
            beam_values.append(new_values)
            tmp_beam_values.append(deepcopy(new_values))
            idx = len(_saved_beam_values)
            _saved_beam_values.append(deepcopy(new_values))
            _saved_beam_values[-1].update({"stamp":f"{distribution}-traverse_order_index_{traverse_order_index}-beam_index_{idx}"})
        tmp_saved_beam_values = deepcopy(tmp_beam_values)
        for i,_ in enumerate(tmp_saved_beam_values):
            _.update({"stamp":f"{distribution}-traverse_order_index_{traverse_order_index}-beam_index_{i}"})

        (tmp_correct_beam_values, tmp_original_beam_indexs, tmp_saved_beam_values), verify_output_cache, accepted_num, factual_dynamic_start_threshold, change_flag = self._verify(rules_to_search, tmp_beam_values, var, traverse_order_index, get_verifier_samples,
                                                                                    datatype_verifier_samples, factual_verifier_samples, tmp_saved_beam_values,
                                                                                    datatype_verifier_threshold, factual_verifier_threshold,
                                                                                    dynamic_verifier_threshold, datatype_lowest_verifier_threshold, factual_lowest_verifier_threshold, dynamic_threshold_step,
                                                                                    factual_dynamic_start_threshold,
                                                                                    verify_output_cache, accumulate_verifier_confidence, weighted_accumulate)
        correct_beam_values.extend(tmp_correct_beam_values)
        for i, idx in enumerate(tmp_original_beam_indexs):
            original_beam_indexs.append(previous_idx+idx)
        correct_values = []
        wrong_values = []
        for i, _ in enumerate(tmp_saved_beam_values):
            _saved_beam_values[previous_idx+i]["datatype_ver"] = _["datatype_ver"]
            _saved_beam_values[previous_idx+i]["factual_ver"] = _["factual_ver"]
            if _["stamp"].endswith("-out_verifier_datatype"):
                _saved_beam_values[previous_idx+i]["stamp"] += "-out_verifier_datatype"
                wrong_values.append(_[var])
            elif _["stamp"].endswith("-out_verifier_factual"):
                _saved_beam_values[previous_idx+i]["stamp"] += "-out_verifier_factual"
                wrong_values.append(_[var])
            else:
                correct_values.append(_[var])
        return correct_beam_values, original_beam_indexs, _saved_beam_values, beam_values, correct_values, wrong_values, verify_output_cache, accepted_num, factual_dynamic_start_threshold, change_flag

    def _verify_factual_node(self, beam_values, variable_values, variable_values_2, values,
                             distribution, traverse_order_index, var,
                             correct_beam_values, original_beam_indexs, _saved_beam_values,
                             verify_output_cache):
        """
        verify search results of factual node
        """
        accept = False
        if len(variable_values) and len(variable_values_2) and variable_values[0] == variable_values_2[0]:
                # answers are the same, correct
                accept = True
        if accept:
            # accept the value
            previous_idx = len(beam_values)
            factual_value = variable_values[0]
            new_values = deepcopy(values)
            new_values[var] = factual_value
            tmp_beam_values = [deepcopy(new_values)]
            # update beam_values
            beam_values.append(deepcopy(new_values))
            # update _saved_beam_values
            idx = len(_saved_beam_values)
            _saved_beam_values.append(deepcopy(new_values))
            _saved_beam_values[-1].update({"stamp":f"{distribution}-traverse_order_index_{traverse_order_index}-beam_index_{idx}"})
            # update correct_beam_values
            correct_beam_values.extend(tmp_beam_values)
            # update original_beam_indexs
            original_beam_indexs.append(previous_idx+0)
            # update verify_output_cache
            value_str = "_".join(new_values.values())
            new_values.pop(var)
            previous_value = "_".join(new_values.values())
            verify_output_cache[value_str] = verify_output_cache[previous_value]
        return correct_beam_values, original_beam_indexs, _saved_beam_values, beam_values, verify_output_cache


def diverse_prompt_format(premise, conclusion, collect_rationale=False, cot=False):
    """
    generate diverse templates for probing
    """
    # premise: Tom was born in the Roman Empire. conclusion: Tom cannot use laptop.
    # premise: Tom is allergic to nuts and does not use multi-grain bread but uses wheat bread in his PB&J. conclusion: Tom cannot eat the PB&J.
    def convert_to_question(conclusion):
        prompt = f"Convert the statement to a question with minimal edits: {conclusion}"
        GPT3Model = OpenAIWrapper("text-davinci-003")
        output = GPT3Model.generate([prompt], 1, 0.9, 32)[0][0]
        return output.strip()
    
    def negate_statement(conclusion):
        prompt = f"Negate the statement with minimal edits: {conclusion}"
        GPT3Model = OpenAIWrapper("text-davinci-003")
        output = GPT3Model.generate([prompt], 1, 0.9, 32)[0][0]
        return output.strip()
    
    def complete_sentence(sentence):
        if sentence.strip().endswith("."):
            return sentence
        else:
            return sentence + "."
    conclusion = complete_sentence(conclusion)
    conclusion_question = convert_to_question(conclusion)
    conclusion_negation = negate_statement(conclusion)
    conclusion_negation = complete_sentence(conclusion_negation)
    if not cot:
        if collect_rationale:
            positive_label_formats = [
                (f"Is it true that if {premise}, {conclusion[:-1]}? Answer yes or no, and then give an explanation to your answer:", "Yes"),
                (f"Yes or no: if {premise}, {conclusion} Also give an explanation to the answer.", "Yes"),
                (f"True or false: if {premise}, {conclusion} Also give an explanation to the answer.", "True"),
                (f"Right or Wrong: if {premise}, {conclusion} Also give an explanation to the answer.", "Right"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion? Answer yes or no, and then give an explanation to your answer:", "Yes"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion? Answer yes or no, and then give an explanation to your answer:", "Yes"),
            ]
            negative_label_formats = [
                (f"Is it true that if {premise}, {conclusion_negation[:-1]}? Answer yes or no, and then give an explanation to your answer:", "No"),
                (f"Yes or no: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "No"),
                (f"Answer the question with yes or no, and then give an explanation to your answer: if {premise}, {conclusion_question}", "No"),
                (f"True or false: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "False"),
                (f"Right or Wrong: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "Wrong"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion? Answer yes or no, and then give an explanation to your answer:", "No"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion? Answer yes or no, and then give an explanation to your answer:", "No"),
            ]
        else:
            positive_label_formats = [
                (f"Is it true that if {premise}, {conclusion[:-1]}? Answer yes or no:", "Yes"),
                (f"Yes or no: if {premise}, {conclusion}", "Yes"),
                (f"True or false: if {premise}, {conclusion}", "True"),
                (f"Right or Wrong: if {premise}, {conclusion}", "Right"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion? Answer yes or no:", "Yes"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion? Answer yes or no:", "Yes"),
            ]
            negative_label_formats = [
                (f"Is it true that if {premise}, {conclusion_negation[:-1]}? Answer yes or no:", "No"),
                (f"Yes or no: if {premise}, {conclusion_negation}", "No"),
                (f"Answer the question with yes or no: if {premise}, {conclusion_question}", "No"),
                (f"True or false: if {premise}, {conclusion_negation}", "False"),
                (f"Right or Wrong: if {premise}, {conclusion_negation}", "Wrong"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion? Answer yes or no:", "No"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion? Answer yes or no:", "No"),
            ]
    else:
        positive_label_formats = [
            (f"Is it true that if {premise}, {conclusion[:-1]}?", "Therefore, the answer (yes or no) is:", "Yes"),
            (f"Yes or no: if {premise}, {conclusion}", "Therefore, the answer (yes or no) is:", "Yes"),
            (f"True or false: if {premise}, {conclusion}", "Therefore, the answer (true or false) is:", "True"),
            (f"Right or Wrong: if {premise}, {conclusion}", "Therefore, the answer (right or wrong) is:", "Right"),
            (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion?", "Therefore, the answer (yes or no) is:", "Yes"),
            (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion?", "Therefore, the answer (yes or no) is:", "Yes"),
        ]
        negative_label_formats = [
            (f"Is it true that if {premise}, {conclusion_negation[:-1]}?", "Therefore, the answer (yes or no) is:", "No"),
            (f"Yes or no: if {premise}, {conclusion_negation}", "Therefore, the answer (yes or no) is:", "No"),
            (f"Answer the question: if {premise}, {conclusion_question}", "Therefore, the answer (yes or no) is:", "No"),
            (f"True or false: if {premise}, {conclusion_negation}", "Therefore, the answer (true or false) is:", "False"),
            (f"Right or Wrong: if {premise}, {conclusion_negation}", "Therefore, the answer (right or wrong) is:", "Wrong"),
            (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion?", "Therefore, the answer (yes or no) is:", "No"),
            (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion?", "Therefore, the answer (yes or no) is:", "No"),
        ]
    
    positive_label_formats = [(i[0].replace("if if ","if "),*i[1:]) for i in positive_label_formats]
    negative_label_formats = [(i[0].replace("if if ","if "),*i[1:]) for i in negative_label_formats]

    return range(len(positive_label_formats) + len(negative_label_formats)), positive_label_formats + negative_label_formats


class OpenAIWrapper:
    def __init__(self, model_path):
        self.model_path = model_path

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError))
    def get_logprob(self,prompts,max_tokens=0, get_last = False):
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
            if not get_last:
                score = np.mean(response["choices"][0]["logprobs"]['token_logprobs'][1:])
            else:
                score = np.mean(response["choices"][0]["logprobs"]['token_logprobs'][-1])
            scores.append(score)
        return scores

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError))
    def generate(self, prompts, num_sample, temperature, max_tokens, logit_bias={}):
        outputs = []
        for prompt in tqdm(prompts):
            if self.model_path == "text-davinci-003":
                texts = []
                while True:
                    try:
                        response = openai.Completion.create(
                            engine=self.model_path,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=1.0,
                            n=num_sample,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            logit_bias=logit_bias
                        )
                        break
                    except KeyboardInterrupt:
                        assert 0
                    except Exception as e:
                        # try again
                        logger.info("try again")
                logger.info("Prompt: {}\n".format(prompt))
                for i, choice in enumerate(response["choices"]):
                    text = choice['text']
                    logger.info("Response text: {}".format(text))
                    texts.append(text)
                outputs.append(texts)
            elif self.model_path == "gpt-3.5-turbo" or self.model_path == "gpt-4" or self.model_path == "gpt-3.5-turbo-16k" or self.model_path == "gpt-4-32k-0613":
                texts = []
                while True:
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.model_path,
                            messages=[{"role": 'user', "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=1.0,
                            n=num_sample,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            logit_bias=logit_bias
                        )
                        break
                    except KeyboardInterrupt:
                        assert 0
                    except Exception as e:
                        logger.info("try again")
                for i, choice in enumerate(response["choices"]):
                    text = choice['message']['content']
                    texts.append(text)
                outputs.append(texts)
            else:
                raise NotImplementedError
        return outputs


class KnowledgeModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        if model_path == "text-davinci-003" or model_path == "gpt-3.5-turbo" or model_path == "gpt-4":
            self.model = OpenAIWrapper(model_path)
            self.tokenizer = None
            self.tiktoken = tiktoken.encoding_for_model(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.parallelize()
            for device in range(torch.cuda.device_count()):
                logger.info("device: %d, name: %s", device, torch.cuda.get_device_name(device))
            self.model.eval()
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, inputs, num_sample=1, temperature=0.9, max_tokens=16, logit_bias={}):
        """Given inputs string, generate a list (num_inputs) of list (num_sample) of strings"""
        if self.tokenizer is not None:
            input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
            output_ids = self.beam_search(input_ids, num_beams=num_sample, temperature=temperature, max_length=max_tokens) # fix this
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            outputs = self.model.generate(inputs, num_sample, temperature, max_tokens, logit_bias)
        return outputs

    def post_process(self, output, var=None, datatype=None):
        """postprocess to get the values from model output"""
        # # output: \n\n1. Google Hangouts. \n2. Microsoft Word. \n3. Apple iPhone.
        output_values = []
        outputs = output.split("\n")
        for instance in outputs:
            instance = instance.strip()
            if instance != "":
                if "." not in instance or instance[-1] != '.':
                    continue
                value = instance[instance.index(".")+1:-1].strip("'. ").strip('" ')
                if "/" in value or value.count(",") > 1 or len(value) > 64:
                    # exclude some extremely long values
                    continue
                if "," in value:
                    # keep the first part before comma
                    value = value[:value.index(",")].strip()
                if "(" in value and ")" in value:
                    # exclude content in the parenthesis
                    value = value[:value.index("(")] + value[value.index(")")+1:].strip()
                if ":" in value:
                    # e.g., Technology: Virtual Reality -> Virtual Reality
                    value = value[value.index(":")+1:].strip()
                if " - " in value:
                    # e.g., Technology - Virtual Reality -> Virtual Reality
                    value = value[value.index(" - ")+3:].strip()
                value = " ".join(value.split("_")).strip()
                if ". " in value:
                    # filter out cases like "Printing. Was invented in 1984"
                    continue
                # check whether the value containing var or datatype
                split_value = value.split()
                if var in split_value or value == datatype:
                    continue
                if value != "":
                    output_values.append(value)
        return output_values

    def construct_prompt(self, var, rules, values, datatype, nodetype, beam_size, premise_map={}, conclusion_map={}, nl_prompt=False):
        """
        var: string (e.g. "B")
        rules: list of strings (e.g. ['exist(B, D)', 'is_used_on(C, B}'])
        values: dict of string to string (e.g. {'B': 'lawn', 'C': 'leaf blower', 'D': 'Renaissance'}) all previously searched values
        """
        if nl_prompt:
            nl_processed_rules = []
            for r in rules:
                nl_r = premise_map[r] if r in premise_map else conclusion_map[r]
                nodes = re.findall(r"\[[A-Z]\]", nl_r)
                for n in nodes:
                    if n.strip("[]") in values:
                        nl_r = nl_r.replace(n, values[n.strip("[]")])
                nl_processed_rules.append(nl_r)
            nl_rule = " and ".join(nl_processed_rules)
            if nodetype == "factual":
                template = f"Give me the correct value of {var} to fill in the sentence '{nl_rule}' in the format '1. value.', where {var} is a {datatype.lower()}."
            else:
                template = f"Give me {beam_size} values of {var} to fill in the sentence '{nl_rule}' in the format '1. value.', where {var} is a {datatype.lower()}."
        else:
            processed_rules = []
            for r in rules:
                nodes = r[r.index("(")+1: r.index(")")].split(",")
                for n in nodes:
                    if n.strip() in values:
                        r = r.replace(n.strip(), values[n.strip()])
                processed_rules.append(r)
            rule = " & ".join(processed_rules)
            if nodetype == "factual":
                template = f"Give me the correct value of {var} to instantiate the rule '{rule}' in the format '1. value.', where {var} is a {datatype}."
            else:
                template = f"Give me {beam_size} diverse values of {var} to instantiate the rule '{rule}' in the format '1. value.', where {var} can be a {datatype}."
        return template


class LanguageModel(torch.nn.Module):
    def __init__(self, model_path,use_ft = False):
        super().__init__()
        self.model_path = model_path
        assert model_path != "gpt-3.5-turbo" or model_path != "gpt-4"
        if model_path == "text-davinci-003":
            self.model = OpenAIWrapper(model_path)
            self.tokenizer = None
        else:
            if 'llama' in model_path:
                DEFAULT_PAD_TOKEN = "[PAD]"
                DEFAULT_EOS_TOKEN = "</s>"
                DEFAULT_BOS_TOKEN = "<s>"
                DEFAULT_UNK_TOKEN = "</s>"
                # use local_files_only = True to load model from cache
                self.tokenizer = LlamaTokenizer.from_pretrained(model_path,padding_side = "left", local_files_only=True)
                self.tokenizer.add_special_tokens(
                    {
                        "eos_token": DEFAULT_EOS_TOKEN,
                        "bos_token": DEFAULT_BOS_TOKEN,
                        "unk_token": DEFAULT_UNK_TOKEN,
                        "pad_token": DEFAULT_PAD_TOKEN,
                    })
                self.model = LlamaForCausalLM.from_pretrained(model_path,device_map = "auto",torch_dtype = torch.float16, local_files_only=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map= 'auto')
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            for device in range(torch.cuda.device_count()):
                logger.info("language_model device: %d, name: %s", device, torch.cuda.get_device_name(device))
            self.model.eval()

        if use_ft:
            import fasttext
            self.ft = fasttext.load_model('./fasttext/crawl-300d-2M-subword.bin')

    def calculate_likelihood(self, inputs, fasttext = False, batch_size = 32):
        """calculate the sentence likelihood"""
        assert self.model_path != "gpt-3.5-turbo"
        
        if not fasttext:
            if self.model_path == "text-davinci-003":
                return self.model.get_logprob(inputs)
            else:
                with torch.no_grad():
                    scores_l = []
                    for batch in batch_process(inputs,batch_size):
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
        else:
            assert isinstance(inputs, list)
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


class VerifierModel(torch.nn.Module):
    
    def __init__(self, model_path, positive_token, negative_token):
        super().__init__()
        self.model_path = model_path
        self.positive_token = positive_token
        self.negative_token = negative_token
        if model_path == "text-davinci-003" or model_path == "gpt-3.5-turbo":
            self.model = OpenAIWrapper(model_path)
            self.tokenizer = None
        else:
            if 'llama' in model_path:
                DEFAULT_PAD_TOKEN = "[PAD]"
                DEFAULT_EOS_TOKEN = "</s>"
                DEFAULT_BOS_TOKEN = "</s>"
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
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map= 'auto')
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            for device in range(torch.cuda.device_count()):
                logger.info("verifier_model device: %d, name: %s", device, torch.cuda.get_device_name(device))
            self.model.eval()
            self.positive_index, self.negative_index = self._get_label_index()
            logger.info(f"self.positive_index: {self.positive_index}")
            logger.info(f"self.negative_index: {self.negative_index}")

    def verify(self, inputs_ll, get_verifier_samples=True, verifier_threshold=0.2, batch_size = 32, previous_confidence=0, accumulate_verifier_confidence=False, weighted_accumulate=False):
        """verify the values with given threshold"""
        logger.info(f"inputs_ll: {inputs_ll}")
        assert len(inputs_ll) == len(previous_confidence)
        if self.model_path == "text-davinci-003" or self.model_path == "gpt-3.5-turbo":
            outputs = []
            verifier_samples = []
            if not any([len(_) for _ in inputs_ll]):
                for _ in inputs_ll:
                    outputs.append(self.positive_token)
            else:
                yes_l= [_[0] + " Yes" for _ in inputs_ll]
                no_l= [_[0] + " No" for _ in inputs_ll]
                log_probs_yes = self.model.get_logprob(yes_l, get_last = True)
                log_probs_no = self.model.get_logprob(no_l, get_last = True)
                outputs = [self.positive_token if log_probs_yes[i] > log_probs_no[i] else self.negative_token for i in range(len(log_probs_yes))]
                for i in range(len(log_probs_yes)):
                    softmax_probs = F.softmax(torch.Tensor([log_probs_yes[i], log_probs_no[i]]), dim=-1)
                    accumulated_probs_yes_l = previous_confidence[i][0] + [softmax_probs[0].item()]
                    accumulated_probs_no_l = previous_confidence[i][1]+ [softmax_probs[1].item()]
                    if weighted_accumulate:
                        raise NotImplementedError
                    else:
                        accumulated_probs_yes = np.mean(accumulated_probs_yes_l)
                        accumulated_probs_no = np.mean(accumulated_probs_no_l)
                    logger.info("accumulated_probs_yes_l: {}, accumulated_probs_yes: {}".format(accumulated_probs_yes_l, accumulated_probs_yes))
                    logger.info("accumulated_probs_no_l: {}, accumulated_probs_no: {}".format(accumulated_probs_no_l, accumulated_probs_no))
                    if accumulate_verifier_confidence:
                        if accumulated_probs_yes < verifier_threshold:
                            outputs[i] = self.negative_token
                    else:
                        if softmax_probs[0].item() < verifier_threshold:
                            outputs[i] = self.negative_token
                    verifier_samples.append({"input_text": inputs_ll[i][0], "verifier_output": (softmax_probs[0].item(), softmax_probs[1].item()), "raw_verifier_output": (log_probs_yes[i], log_probs_no[i]), "accumulated_output": (accumulated_probs_yes, accumulated_probs_no), "accumulated_output_list": (accumulated_probs_yes_l, accumulated_probs_no_l), "previous_accumulated_output": previous_confidence[i], "verifier_threshold": verifier_threshold})
            return (outputs, verifier_samples) if get_verifier_samples else (outputs, [])
        
        elif "t5" in self.model_path:
            all_choices = []
            verifier_samples = []
            with torch.no_grad():
                if not any([len(_) for _ in inputs_ll]):
                    for _ in inputs_ll:
                        all_choices.append(self.positive_token)
                else:

                    for _inputs_ll, _previous_confidence in zip(batch_process(inputs_ll,batch_size), batch_process(previous_confidence, batch_size)):
                            
                        input_texts = [b for a in _inputs_ll for b in a ]
                        input_ids = self.tokenizer(input_texts, padding=True, return_tensors="pt").to('cuda')
                        decoder_input_ids = torch.tensor([self.tokenizer.pad_token_id], device=torch.cuda.current_device()).unsqueeze(0)
                        decoder_input_ids = decoder_input_ids.expand(len(input_texts), -1)
                        outputs = self.model(**input_ids, decoder_input_ids=decoder_input_ids)
                        logits = outputs.logits
                        selected_logits = logits[list(range(logits.shape[0])),-1][:,[self.positive_index, self.negative_index]].unsqueeze(1)
                        selected_logits = selected_logits.squeeze(1)
                        logger.info("selected_logits: {}".format(selected_logits))
                        for logit_pair in selected_logits:
                            if logit_pair[0] == logit_pair[1]:
                                import pdb; pdb.set_trace()
                        for i, c in enumerate(selected_logits):
                            corrects = True
                            softmax_probs = F.softmax(c, dim=-1)
                            accumulated_probs_yes_l = _previous_confidence[i][0] + [softmax_probs[0].item()]
                            accumulated_probs_no_l = _previous_confidence[i][1]+ [softmax_probs[1].item()]
                            if weighted_accumulate:
                                raise NotImplementedError
                            else:
                                accumulated_probs_yes = np.mean(accumulated_probs_yes_l)
                                accumulated_probs_no = np.mean(accumulated_probs_no_l)
                            logger.info("accumulated_probs_yes_l: {}, accumulated_probs_yes: {}".format(accumulated_probs_yes_l, accumulated_probs_yes))
                            logger.info("accumulated_probs_no_l: {}, accumulated_probs_no: {}".format(accumulated_probs_no_l, accumulated_probs_no))
                            verifier_samples.append({"input_text": input_texts[i], "verifier_output": (softmax_probs[0].item(), softmax_probs[1].item()), "raw_verifier_output": (c[0].item(), c[1].item()), "accumulated_output": (accumulated_probs_yes, accumulated_probs_no), "accumulated_output_list": (accumulated_probs_yes_l, accumulated_probs_no_l), "previous_accumulated_output": _previous_confidence[i], "verifier_threshold": verifier_threshold})
                            if accumulate_verifier_confidence:
                                if accumulated_probs_yes < verifier_threshold:
                                    corrects = False
                            else:
                                if softmax_probs[0].item() < verifier_threshold:
                                    corrects = False
                            if corrects:
                                all_choices.append(self.positive_token)
                            else:
                                all_choices.append(self.negative_token)

            return (all_choices, verifier_samples) if get_verifier_samples else (all_choices, [])

    def _get_label_index(self):
        # yes, no, true, false, correct, incorrect, right, wrong
        positive_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{self.positive_token}"))[0]
        negative_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{self.negative_token}"))[0]
        return positive_index, negative_index


class ProbingModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        if model_path in ["text-davinci-003","gpt-3.5-turbo","gpt-4"]:
            self.model = OpenAIWrapper(model_path)
            self.tokenizer = None
        else:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map= 'auto')
            for device in range(torch.cuda.device_count()):
                logger.info("probing_model device: %d, name: %s", device, torch.cuda.get_device_name(device))
            self.model.eval()
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def probe(self, inputs, cot_answer_inputs, labels, indices, collect_rationale=False, cot=False, cot_icl = False):
        """
        Do entailment classification task; return the correct results, wrong results and the overall accuracy
        """
        correct = []
        wrong = []
        if collect_rationale:
            rationale = []
        if self.tokenizer is not None:
            for i, input in enumerate(inputs):
                input_ids = self.tokenizer(input, return_tensors="pt").input_ids
                input_ids = torch.tensor(input_ids, device=torch.cuda.current_device())
                output_ids = self.model.generate(input_ids, max_length=32)
                token = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True) # this is assuming the dimension is (batch_size, seq_len) and sequence starts with <s>

                if token == labels[i]:
                    correct.append((indices[i], inputs[i], token))
                else:
                    wrong.append((indices[i], inputs[i], token))
        else:
            if cot:

                if cot_icl:
                    outputs = []
                    for input in tqdm(inputs,desc="probe w cot icl"):
                        output = self.model.generate([input], num_sample=1, temperature=0.9, max_tokens=256 if collect_rationale else 16)
                        outputs.extend(output)
                    rationale = [output[0].strip() for output in outputs]
                    rationale = [_.split("\nTherefore")[0] for _ in rationale]
                    answer_inputs = [input + '\n' + r + '\n' + cot_answer_input for input, r, cot_answer_input in zip(inputs, rationale, cot_answer_inputs)]
                    outputs = self.model.generate(answer_inputs, num_sample=1, temperature=0.9, max_tokens=16)
                    answers = [word_tokenize(output[0].strip())[0] for output in outputs]
                else:
                    outputs = []
                    for input in tqdm(inputs,desc="probe w cot only"):
                        output = self.model.generate([input], num_sample=1, temperature=0.9, max_tokens=256 if collect_rationale else 16)
                        outputs.extend(output)
                    rationale = [output[0].strip() for output in outputs]
                    # logger.info("rationale example {}".format(rationale[:5]))
                    answer_inputs = [input + '\n' + r + '\n' + cot_answer_input for input, r, cot_answer_input in zip(inputs, rationale, cot_answer_inputs)]
                    outputs = self.model.generate(answer_inputs, num_sample=1, temperature=0.9, max_tokens=16)
                    answers = [word_tokenize(output[0].strip())[0] for output in outputs]
                for i, token in enumerate(answers):
                    # logger.info("token: %s", token)
                    if token.lower() == labels[i].lower():
                        if collect_rationale:
                            correct.append((indices[i], answer_inputs[i], token, rationale[i]))
                        else:
                            correct.append((indices[i], answer_inputs[i], token))
                    else:
                        if collect_rationale:
                            wrong.append((indices[i], answer_inputs[i], token, rationale[i]))
                        else:
                            wrong.append((indices[i], answer_inputs[i], token))
            else:
                outputs = self.model.generate(inputs, num_sample=1, temperature=0.9, max_tokens=256 if collect_rationale else 16)
                # print("outputs:", outputs)
                tokens = [word_tokenize(output[0].strip()) for output in outputs] # this is assuming the dimension is (batch_size, num_sample, seq_len) and sequence starts with the answer
                answers = [token[0] for token in tokens]
                # rationale
                if collect_rationale:
                    for i, (output, token) in enumerate(zip(outputs, tokens)):
                        output = output[0].strip()
                        if len(token) > 2:
                            answer_len = len("".join(token[:2]))
                            rationale.append(output[answer_len:].strip())
                        else:
                            rationale.append("")
                # answer
                for i, token in enumerate(answers):
                    # logger.info("token: %s", token)
                    if token.lower() == labels[i].lower():
                        if collect_rationale:
                            correct.append((indices[i], inputs[i], token, rationale[i]))
                        else:
                            correct.append((indices[i], inputs[i], token))
                    else:
                        if collect_rationale:
                            wrong.append((indices[i], inputs[i], token, rationale[i]))
                        else:
                            wrong.append((indices[i], inputs[i], token))
        if len(correct) + len(wrong):
            accuracy  = len(correct) / (len(correct) + len(wrong))
        else:
            accuracy = 0
        return correct, wrong, accuracy


def run_beam_search(knowledge_model, likelihood_model, verifier_model, rule, graph, traverse_order, args, output_file, verifier_safe_file, saved_beam_file, verify_accumulated_file, traverse_order_entity, rank_as_pre_con):
    """
    run knowledge beam search; return the searched beams
    """
    beam_size, n_sample = args.beam_size, args.knowledge_n_sample
    use_ft = args.rank_by_ft
    get_verifier_samples = args.get_verifier_samples
    deduplicate = args.deduplicate
    bias_value = args.bias_value
    factual_verifier_threshold = args.factual_verifier_threshold
    datatype_verifier_threshold = args.datatype_verifier_threshold
    dynamic_verifier_threshold = args.dynamic_verifier_threshold
    factual_lowest_verifier_threshold = args.factual_lowest_verifier_threshold
    datatype_lowest_verifier_threshold = args.datatype_lowest_verifier_threshold
    dynamic_threshold_step = args.dynamic_threshold_step
    accumulate_verifier_confidence = args.accumulate_verifier_confidence
    nl_prompt = args.nl_prompt
    weighted_accumulate = args.weighted_accumulate
    dynamic_ranker = args.dynamic_ranker
    ranker_accept_ratio = args.ranker_accept_ratio

    """Calls beam search of a rule on a pre-traverse graph and search variables according to traverse_order, and writes the results to a file."""
    bs_generator = BeamSearchGenerator(knowledge_model, likelihood_model, verifier_model, rule, graph, traverse_order, beam_size, traverse_order_entity, rank_as_pre_con)
    head_final_beams, longtail_final_beams, factual_verifier_samples, datatype_verifier_samples, saved_beam_values, verify_output_cache = bs_generator.search(n_sample, use_ft,
                                                                                   get_verifier_samples=get_verifier_samples,
                                                                                   deduplicate=deduplicate,
                                                                                   bias_value=bias_value,
                                                                                   factual_verifier_threshold=factual_verifier_threshold,
                                                                                   datatype_verifier_threshold=datatype_verifier_threshold,
                                                                                   dynamic_verifier_threshold=dynamic_verifier_threshold,
                                                                                   factual_lowest_verifier_threshold=factual_lowest_verifier_threshold,
                                                                                   datatype_lowest_verifier_threshold=datatype_lowest_verifier_threshold,
                                                                                   dynamic_threshold_step=dynamic_threshold_step,
                                                                                   accumulate_verifier_confidence=accumulate_verifier_confidence,
                                                                                   weighted_accumulate=weighted_accumulate,
                                                                                   use_nl_prompt=nl_prompt,
                                                                                   dynamic_ranker=dynamic_ranker,
                                                                                   ranker_accept_ratio=ranker_accept_ratio)
    print("head_final_beams: {}".format(head_final_beams))
    print("longtail_final_beams: {}".format(longtail_final_beams))

    with open(output_file + "_head", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=traverse_order)
        writer.writeheader()
        writer.writerows(head_final_beams)
    with open(output_file + "_longtail", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=traverse_order)
        writer.writeheader()
        writer.writerows(longtail_final_beams)
    with open(verifier_safe_file + "_factual", "w") as f:
        for s in factual_verifier_samples:
            f.write(json.dumps(s) + "\n")
    with open(verifier_safe_file + "_datatype", "w") as f:
        for s in datatype_verifier_samples:
            f.write(json.dumps(s) + "\n")
    
    with jsonlines.open(saved_beam_file + "_longtail",'w') as f:
        f.write_all(saved_beam_values["tail"])
    with jsonlines.open(saved_beam_file + "_head",'w') as f:
        f.write_all(saved_beam_values["head"])

    with open(verify_accumulated_file, "w") as f:
        json.dump(verify_output_cache, f, indent=2)
    
    return head_final_beams, longtail_final_beams


def load_value_cache(value_cache_file, nodes_to_ignore):
    """Load pre-searched value from file using csv.DictReader"""
    with open(value_cache_file + "_head", "r") as f:
        data = pd.read_csv(f, dtype=str)
        data = data.drop(columns=nodes_to_ignore)
        data.drop_duplicates(inplace=True, ignore_index=True)
        head_value_cache = data.to_dict('records')
        # reader = csv.DictReader(f, delimiter=',')
        # head_value_cache = []
        # for row in reader:
        #     head_value_cache.append(row)
    with open(value_cache_file + "_longtail", "r") as f:
        data = pd.read_csv(f, dtype=str)
        data = data.drop(columns=nodes_to_ignore)
        data.drop_duplicates(inplace=True, ignore_index=True)
        longtail_value_cache = data.to_dict('records')
        # reader = csv.DictReader(f, delimiter=',')
        # longtail_value_cache = []
        # for row in reader:
        #     longtail_value_cache.append(row)
    return head_value_cache, longtail_value_cache


def get_rule(rule_key, rule_file):
    """Get rule from rule dictionary. Read from file."""
    rule_dict = json.load(open(rule_file))
    premise_map = rule_dict[rule_key]["premise_map"]
    conclusion_map = rule_dict[rule_key]["conclusion_map"]
    variables_dict = rule_dict[rule_key]["variables"]
    variables = {}
    for key in variables_dict:
        variables[key] = Variable(*variables_dict[key])
    rule = Rule(premise_map=premise_map, conclusion_map=conclusion_map, variables=variables)
    graph, information_edges, constraint_edges, conclusion_first_order, premise_first_order = rule.parse_rule()
    premise_template = " and ".join([rule.premise_map[edge] for edge in information_edges])
    conclusion_template = " and ".join([rule.conclusion_map[edge] for edge in rule.conclusion])
    return rule, graph, conclusion_first_order, premise_first_order, premise_template, conclusion_template


def create_entities_pairs(beam_values,exclude_key):
    l = []
    for value in beam_values:
        value_l = [item for key,item in value.items() if key != exclude_key]
        l.append((value_l))
    return l


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--beam_size", type=int, default=40)
    parser.add_argument("--output_directory", type=str, default="to_be_replaced")
    parser.add_argument("--knowledge_n_sample", type=int, default=10)
    parser.add_argument("--knowledge_model_path", type=str, default="text-davinci-003")
    parser.add_argument("--likelihood_model_path", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--rank_by_ft", action = "store_true", help= "Whether or not use fasttext to rank the beams",)
    parser.add_argument("--verifier_model_path", type=str, default="google/flan-t5-xxl")
    parser.add_argument("--get_verifier_samples", action="store_true")
    parser.add_argument("--factual_verifier_threshold", type=float, default=0.85)
    parser.add_argument("--datatype_verifier_threshold", type=float, default=0.85)
    parser.add_argument("--dynamic_verifier_threshold", action="store_true")
    parser.add_argument("--factual_lowest_verifier_threshold", type=float, default=0.65)
    parser.add_argument("--datatype_lowest_verifier_threshold", type=float, default=0.65)
    parser.add_argument("--dynamic_threshold_step", type=float, default=0.05)
    parser.add_argument("--dynamic_ranker", action="store_true")
    parser.add_argument("--ranker_accept_ratio", type=float, default=0.75)
    parser.add_argument("--probe_model_path", type=str, default="text-davinci-003")
    parser.add_argument("--probe_tokenizer_path", type=str, default=None)
    parser.add_argument("--rule_keys", nargs="+", default=["rule1", "rule3", "rule4"])
    # parser.add_argument("--rule_keys", nargs="+", default=["rule1", "rule2", "rule3", "rule4"])
    parser.add_argument("--do_search", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--cot_icl", action="store_true")
    parser.add_argument("--deduplicate", action="store_true", help="Whether or not suppress the likelihood of previous results when searching")
    parser.add_argument("--bias_value", type=int, default=-100)
    parser.add_argument("--batch_size",default= 32, type = int,help="Batch size when model use batch mode, like during verifer and ranker")
    parser.add_argument("--repetition_penalty", default = 3, type = float, help = "weight the repetition penalty during generation")
    parser.add_argument("--collect_rationale", action="store_true", help="whether or not collect rationale when probing")
    parser.add_argument("--traverse_order", type=str, default="conclusion", help="Whether to traverse the graph in conclusion-first or premise-first order", choices=["conclusion", "premise"])
    parser.add_argument("--fout_path", default= "rule_order_{traverse_order}_first.txt")
    parser.add_argument("--which_verifier_first",choices=["factual","datatype"],default = "factual")
    parser.add_argument("--accumulate_verifier_confidence", action="store_true")
    parser.add_argument("--weighted_accumulate", action="store_true")
    parser.add_argument("--rank_as_pre_con",action= "store_true")
    parser.add_argument("--nl_prompt", action="store_true")
    parser.add_argument("--rule_path", type=str, default=None)
    args = parser.parse_args()
    log_path = f"knowledgesearch_{args.rule_keys[0]}-{args.rule_keys[-1]}.log"
    logging.basicConfig(filename=log_path,
                    filemode='a',format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    logger.info(args)

    global nl
    nl = "\n"

    def name_fn(x):
        if "/" in x:
            x = x.rsplit("/")[1]
        return x

    # load models
    if args.do_search:
        knowledge_model = KnowledgeModel(args.knowledge_model_path)
        likelihood_model = LanguageModel(args.likelihood_model_path, args.rank_by_ft)
        verifier_model = VerifierModel(args.verifier_model_path, "yes", "no")
    if args.do_probe:
        probe_model = ProbingModel(args.probe_model_path)

    output_directory_template = args.output_directory + f"/vf_{name_fn(args.verifier_model_path)}_ll_{name_fn(args.likelihood_model_path)}_src_{name_fn(args.knowledge_model_path)}" + f"_use_ft_{args.rank_by_ft}" + "_{traverse_order}_first" + f"_{args.which_verifier_first}_verifier_first" + f"_nl_prompt_{args.nl_prompt}"


    # rule of interest
    for rule_key in args.rule_keys:
        traverse_order_force_conclusion = False
        logger.info(f"We are searching for rule {rule_key}")
        rule, graph, conclusion_first_order, premise_first_order, premise_template, conclusion_template = get_rule(rule_key, args.rule_path)

        if any([1 if _[1].nodetype == "factual" else 0 for _ in rule.variables.items()]):
            traverse_order_force_conclusion = True
            logger.info("Force the traverse order to conclusion here")

        traverse_order_entity = "conclusion" if traverse_order_force_conclusion else "premise"
        # force the rules which have factual node to be conclusion first, otherwise premise first
        traverse_order = conclusion_first_order if traverse_order_entity == "conclusion" else premise_first_order
        output_directory = output_directory_template.format(traverse_order = traverse_order_entity)


        if args.do_search:
            logger.info("Output dir is {}".format(output_directory))
            pathlib.Path(output_directory).mkdir(exist_ok= True, parents= True)

            value_cache_file = f"{output_directory}/{rule_key}_value_cache.csv"
            verifier_safe_file = f"{output_directory}/{rule_key}_verifier_result.csv"
            saved_beam_file = f"{output_directory}/{rule_key}_saved_beams.json"
            flag_file = f"{output_directory}/{rule_key}_flag"
            verify_accumulated_file = f"{output_directory}/{rule_key}_verify_acculated_prob.json"


            if os.path.exists(f"{value_cache_file}_head"):
                logger.info("Already exist")
                continue

            if os.path.exists(flag_file):
                logger.info("flag file exist and means one of the processing are handling or handled this rule")
                continue

            # no program is doing it, so let's do it and mark this rule as being searched now
            with open(flag_file,"w") as f:
                f.write("1")

            fout = open(output_directory + "/" + args.fout_path.format(traverse_order = args.traverse_order),"a")
            fout.write(nl)
            fout.write(f"{rule_key} {traverse_order_entity}")
            fout.write(nl)

            try:
                head_final_beams, longtail_final_beams = run_beam_search(knowledge_model, likelihood_model, verifier_model, rule, graph, traverse_order, args, value_cache_file, verifier_safe_file,saved_beam_file, verify_accumulated_file, traverse_order_entity, args.rank_as_pre_con)
            except Exception as e:
                logger.error("Error here happen: %s",e,exc_info= True)
                os.remove(flag_file)
                continue

        else:
            if args.input_directory_probe is not None:
                value_cache_file = f"{args.input_directory_probe}/{rule_key}_value_cache.csv"
            else:
                value_cache_file = f"{output_directory}/{rule_key}_value_cache.csv"
            head_final_beams, longtail_final_beams = load_value_cache(value_cache_file, [])

        logger.info("Head final beams: {}".format(head_final_beams))
        logger.info("Longtail final beams: {}".format(longtail_final_beams))
