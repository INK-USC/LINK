import openai
import backoff
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY") 


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
                logger.info("Prompt: {}\n".format(prompt))
                for i, choice in enumerate(response["choices"]):
                    text = choice['text']
                    logger.info("Response text: {}".format(text))
                    texts.append(text)
                outputs.append(texts)
            elif self.model_path == "gpt-3.5-turbo" or self.model_path == "gpt-4":
                texts = []
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
                print("Prompt: {}\n".format(prompt))
                for i, choice in enumerate(response["choices"]):
                    text = choice['message']['content']
                    print("Response text: {}".format(text))
                    texts.append(text)
                outputs.append(texts)
            else:
                raise NotImplementedError
        return outputs


class MetaRule:
    def __init__(self, text):
        self.text = text
        self.premise = text.split(":-")[1]
        self.conclusion = text.split(":-")[0]
        self.premise_list = [premise.strip() for premise in self.premise.split("&")]
        self.variable_to_predicate_map = {}
        self.get_variable_to_predicate_map()

    def __str__(self):
        return self.text
    
    def get_variable_to_predicate_map(self):
        if len(self.variable_to_predicate_map) > 0:
            return self.variable_to_predicate_map
        for predicate in self.premise_list + [self.conclusion]:
            variables = predicate.split("(")[1].split(")")[0]
            for variable in variables.split(","):
                lst = variable.rsplit(" ", 1)
                datatype = lst[0].strip()
                node = lst[1].strip()
                if (node, datatype) not in self.variable_to_predicate_map:
                    self.variable_to_predicate_map[(node, datatype)] = []
                self.variable_to_predicate_map[(node, datatype)].append(predicate)
        return self.variable_to_predicate_map


def read_rules_from_file(file_path):
    """
    read meta rules from the file
    """
    with open(file_path) as f:
        lines = f.readlines()
    rules = []
    for line in lines:
        if " & " in line:
            rule = MetaRule(line.strip())
        else:
            line = line.replace("),", ") &")
            rule = MetaRule(line.strip())
        rules.append(rule)
    return rules


def prompt_expansion(rule, variable, datatype, number, example):
    """
    expand the data type and verbs in the rule
    """
    openai_wrapper = OpenAIWrapper("gpt-4")
    search_rule = " & ".join(rule.variable_to_predicate_map[(variable, datatype)])
    # category_prompt = f'In rule "{search_rule}", {variable} is a variable representing a {datatype}. List {number} subcategories of {datatype} that {variable} could be that also make the rule true.\n\n1. {example}\n2.'
    category_prompt = f'In rule "{search_rule}", {variable} is a variable representing a {datatype}. List {number} subcategories of {datatype} that {variable} could be that also make the rule true.\n\n1.'
    if example != "":
        category_prompt += f" {example}\n2."
    category_output = openai_wrapper.generate([category_prompt], num_sample=1, temperature=0.7, max_tokens=256)[0][0]
    # extract categories from output
    if example != "":
        categories = [example]
    else:
        categories = []
    for line in category_output.split("\n"):
        value = line.split(". ")[-1].strip()
        if value != "":
            categories.append(value)
    all_new_rule_texts = []
    # generate prompts for each category
    for category in categories:
        original_text = rule.text
        for predicate in rule.get_variable_to_predicate_map()[(variable, datatype)]:
            new_predicate = predicate.replace(datatype, category)
            if predicate in rule.conclusion:
                v1 = new_predicate.split("(")[1].split(",")[0]
                v2 = new_predicate.split(",")[1].strip(")").strip()
                nl_predicate = f'{v1} [mask] {v2}'
                verb_prompt = f"Fill in the [mask] in '{nl_predicate}.' so that the statement is factually plausible and grammatically correct. Give 5 statements."
                human_select_verb = "again"
                while human_select_verb == "again":
                    verb_output = openai_wrapper.generate([verb_prompt], num_sample=1, temperature=0.7, max_tokens=256)[0][0]
                    # extract verbs from output
                    verb_output = verb_output.strip().split("\n")
                    verbs = []
                    for o in verb_output:
                        v = o.split(".")[1].strip()
                        verbs.append(v.replace(v1, "").replace(v2, "").strip())
                    print(verbs)
                    human_select_verb = input(f"select a verb: {verbs}")
                new_predicate = f"{human_select_verb}({new_predicate.split('(')[1]}"
            else:
                new_predicate = "[mask](" + new_predicate.split("(")[1]
                # verb_prompt = f"Paraphrase {predicate} to {new_predicate}. Write the best predicate that could fit in [mask] token."
                verb_prompt = f"{predicate} is equal to {new_predicate}. Write the best predicate that could fit in [mask] token."
                # print(verb_prompt)
                verb_output = openai_wrapper.generate([verb_prompt], num_sample=1, temperature=0.7, max_tokens=256)[0][0]
                # extract verbs from output
                verb_output = verb_output.split(". ")[-1].strip()
                verb_output = verb_output.replace(" ", "_").lower()
                new_predicate = new_predicate.replace("[mask]", verb_output)
            original_text = original_text.replace(predicate, new_predicate)
        all_new_rule_texts.append(original_text)
    return all_new_rule_texts


def convert_from_rule_to_natural_language(rule):
    """
    convert symbolic rule to natural language
    """
    predicate = rule.split("(")[0].replace("_", " ").lower()
    variables = rule.split("(")[1].split(")")[0]
    vl = variables.split(",")
    if len(vl) == 2:
        v1 = vl[0].strip()
        v2 = vl[1].strip()
        if v2 != "N":
            new_predicate = f"[{v1}] {predicate} [{v2}]"
        else:
            new_predicate = f"[{v1}] {predicate}"
    else:
        v = vl[0].strip()
        new_predicate = f"[{v}] {predicate}"

    openai_wrapper = OpenAIWrapper("gpt-4")
    convert_nl_prompt = f"Correct the sentence with minimal edits if it is wrong, otherwise repeat it: {new_predicate}."
    cnt = 0
    while cnt < 5:
        # do not change after generating 5 times
        correct_predicate = openai_wrapper.generate([convert_nl_prompt], num_sample=1, temperature=0.9, max_tokens=32)[0][0].strip(".")
        if len(vl) == 2:
            if f"[{v1}]" in correct_predicate and f"[{v2}]" in correct_predicate:
                # follow the format
                new_predicate = correct_predicate
                break
        else:
            if f"[{v}]" in correct_predicate:
                # follow the format
                new_predicate = correct_predicate
                break
        cnt += 1
    return new_predicate


def convert_to_rule_format(all_new_rule_texts, factual_nodes=[], generic_nodes=[]):
    """
    convert the rule from text to our designed dict format
    """
    result_rules = {}
    for i, rule in enumerate(all_new_rule_texts):
        raw_rule = MetaRule(rule)
        all_variables = {}
        # first convert premises
        premise_map = {}
        for predicate in raw_rule.premise_list:
            variables = predicate.split("(")[1].split(")")[0]
            for variable in variables.split(","):
                lst = variable.rsplit(" ", 1)
                datatype = lst[0].strip()
                node = lst[1].strip()
                # first add node to variables
                if node not in all_variables:
                    # all_variables[node] = Variable(node, datatype, False, None)
                    factual = "factual" if node in factual_nodes[i//10] else None
                    if len(generic_nodes) != 0:
                        is_generic = True if node in generic_nodes[i//10] else False
                    else:
                        # default generic node: Person
                        is_generic = True if datatype == "Person" else False
                    all_variables[node] = [
                        node, datatype, is_generic, factual
                    ]
                # then reconstruct predicate
                predicate = predicate.replace(datatype + " ", "")
            nl_predicate = convert_from_rule_to_natural_language(predicate)
            premise_map[predicate] = nl_predicate
        # then convert conclusion
        conclusion_map = {}
        predicate = raw_rule.conclusion
        variables = predicate.split("(")[1].split(")")[0]
        for variable in variables.split(","):
            lst = variable.rsplit(" ", 1)
            datatype = lst[0].strip()
            node = lst[1].strip()
            # first add node to variables
            if node not in all_variables:
                # all_variables[node] = Variable(node, datatype, False, None)
                factual = "factual" if node in factual_nodes else None
                is_generic = True if datatype == "Person" else False
                all_variables[node] = [
                    node, datatype, is_generic, factual
                ]
            # then reconstruct predicate
            predicate = predicate.replace(datatype + " ", "")
        nl_predicate = convert_from_rule_to_natural_language(predicate)
        conclusion_map[predicate] = nl_predicate
        final_rule = dict(
            premise_map = premise_map,
            conclusion_map = conclusion_map,
            variables = all_variables
        )
        result_rules[f"rule{i}"] = final_rule
    return result_rules


def save_to_file(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    rules = [MetaRule("cannot_fit_in(Object A, Bag B) :- has_trouble_containing(Bag B, Electronic Device C) & is_larger_than(Object A, Electronic Device C)")]
    all_new_rule_texts = []
    examples = [
        ""
    ]
    object_variables = [
        "A"
    ]
    factual_nodes = [
        []
    ]
    generic_nodes = [
        ["B"]
    ]
    for i, rule in enumerate(rules):
        print(rule.variable_to_predicate_map)
        all_new_rule_texts.extend(prompt_expansion(rule, object_variables[i], "Object", 10, examples[i]))
        print(all_new_rule_texts)
    result_rules = convert_to_rule_format(all_new_rule_texts, factual_nodes, generic_nodes)

    print(result_rules)
    save_to_file("./data/experimental_rules_object_test_tmp.json", result_rules)
