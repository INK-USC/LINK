from knowledge_beam_search import OpenAIWrapper, get_rule
import torch
import argparse
import logging
import openai
import os
import re
import tiktoken
import csv
import pandas as pd
import pathlib
logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
os.environ["TIKTOKEN_CACHE_DIR"] = ""

openai.api_key = os.getenv("OPENAI_API_KEY") 


class KnowledgeModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = OpenAIWrapper(model_path)
        self.tokenizer = None
        self.tiktoken = tiktoken.encoding_for_model(model_path)

    def generate(self, inputs, num_sample=1, temperature=0.9, max_tokens=16, logit_bias={}):
        """Given inputs string, generate a list (num_inputs) of list (num_sample) of strings"""
        outputs = self.model.generate(inputs, num_sample, temperature, max_tokens, logit_bias)
        return outputs

    def post_process(self, output, nodes):
        """postprocess to get the generated values from model output"""
        output_values = []
        outputs = output.split("\n")
        for instance in outputs:
            instance = instance.strip()
            if instance != "":
                if "." not in instance:
                    continue
                if len(re.findall("=", instance)) < len(nodes):
                    continue
                values = instance[instance.index(".")+1:].strip("'. ").strip('" ')
                value_dict = {}
                split_values = []
                for i in range(len(nodes)):
                    try:
                        try:
                            if i == 0:
                                split_values.append(values[:values.index(f", {nodes[i+1]}=")])
                            elif i == len(nodes)-1:
                                split_values.append(values[values.index(f", {nodes[i]}=")+1:])
                            else:
                                split_values.append(values[values.index(f", {nodes[i]}=")+1:values.index(f", {nodes[i+1]}=")])
                        except:
                            try:
                                if i == 0:
                                    split_values.append(values[:values.index(f", {nodes[i+1]} =")])
                                elif i == len(nodes)-1:
                                    split_values.append(values[values.index(f", {nodes[i]} =")+1:])
                                else:
                                    split_values.append(values[values.index(f", {nodes[i]} =")+1:values.index(f", {nodes[i+1]} =")])
                            except:
                                if i == 0:
                                    split_values.append(values[:values.index(f", {nodes[i+1]}= ")])
                                elif i == len(nodes)-1:
                                    split_values.append(values[values.index(f", {nodes[i]}= ")+1:])
                                else:
                                    split_values.append(values[values.index(f", {nodes[i]}= ")+1:values.index(f", {nodes[i+1]}= ")])
                    except:
                        import sys
                        sys.exit(1)

                for value in split_values:
                    node = value.split("=")[0].strip()
                    val = value.split("=")[1].strip()
                    if node in nodes:
                        value_dict[node] = val
                if len(value_dict) == len(nodes):
                    output_values.append(value_dict)
        return output_values

    def construct_prompt(self, beam_size, rule, premise_template, conclusion_template, generic_nodes, distribution, full=False):
        """construct baseline search prompt"""
        if full:
            # full statement
            premise = " and ".join(rule.premise_map.values())
            conclusion = " and ".join(rule.conclusion_map.values())
        else:
            premise = premise_template
            conclusion = conclusion_template
        # verbalize generic nodes
        for node in generic_nodes:
            # premise  = premise.replace(f"[{node}]", "PersonX") # this is hard coded for rules whose generic node is a person
            # conclusion = conclusion.replace(f"[{node}]", "PersonX")
            premise  = premise.replace(f"[{node}]", f"{rule.variables[node].datatype}X") # this is hard coded for rules whose generic node is a person
            conclusion = conclusion.replace(f"[{node}]", f"{rule.variables[node].datatype}X")
        premise_nodes = re.findall(r'\[(.*?)\]', premise)
        conclusion_nodes = re.findall(r'\[(.*?)\]', conclusion)
        nodes = list(set(premise_nodes + conclusion_nodes))
        nodes.sort()
        datatype_req = [f"{node} is a {rule.variables[node].datatype}" for node in nodes]
        datatype_req = ", ".join(datatype_req)
        nodes_name = " and ".join(nodes)
        format_req = ", ".join([f"{node}=" for node in nodes])
        # only support zero-shot now
        template = f"In the following sentence, {datatype_req}. Find values of {nodes_name} to fill in the blank in the sentence 'If {premise}, then {conclusion}.' and make it a grammatical and correct sentence."
        if distribution == "longtail":
            template += f" Use less frequent terms of {nodes_name}."
            # uncomment templates below to experiment with different templates.
            # template += f" Use terms of {nodes_name} that are less common." # 1
            # template += f" Use terms with lower frequency for {nodes_name}." # 2
            # template += f" Use terms of {nodes_name} that have lower probability in language model distribution." # 3
        template += f" Give me {beam_size} values in the format '1. {format_req}'."
        return template, nodes


def run_search(knowledge_model, rule, premise_template, conclusion_template, args, output_file, distribution, use_logit_bias=True):
    """
    run searching with LLMs
    """
    beam_size = args.beam_size
    n_sample = args.search_n_sample
    deduplicate = args.deduplicate
    bias_value = args.bias_value
    full = args.full
    generic_nodes = []
    for name, var in rule.variables.items():
        if var.is_generic:
            generic_nodes.append(name)

    assert distribution in ['head', 'longtail'], f"{distribution} should be head or longtail!"

    base_prompt, nodes = knowledge_model.construct_prompt(n_sample, rule, premise_template, conclusion_template, generic_nodes, distribution, full)
    logger.info(f"Base Prompt:\n{base_prompt}")

    # search {call_times} times, {n_sample} values once -> {beam_size} values in total
    assert beam_size % n_sample == 0, "beam_size should be the multiple of n_sample"
    call_times = int(beam_size / n_sample)
    final_values = []
    if deduplicate and call_times != 1:
        logit_bias = {}
        outputs = []
        prompt = base_prompt
        duplicate_cnt = 0
        max_tokens = 3072
        if knowledge_model.model.model_path == "gpt-3.5-turbo":
            max_tokens = 2048
        logger.info(f"Max Token: {max_tokens}")
        for i in range(call_times):
            output = knowledge_model.generate([prompt], max_tokens=max_tokens, temperature=0.7, logit_bias=logit_bias)[0][0].strip()
            logger.info(f'Generated Text: \n{output}')
            while output.strip().startswith("1.") is False:
                output = knowledge_model.generate([prompt], max_tokens=max_tokens, temperature=0.7, logit_bias=logit_bias)[0][0].strip()
                logger.info(f'Generated Text: \n{output}')
            outputs.append(output)
            values = knowledge_model.post_process(output, nodes)
            for v in values:
                for node in generic_nodes:
                    # v[node] = "PersonX"
                    v[node] = f"{rule.variables[node].datatype}X"
                if v not in final_values:
                    final_values.append(v)
                else:
                    duplicate_cnt += 1
            if use_logit_bias:
                # add logit_bias when the limit is not reached
                # do not change tokens in template
                template_tokens = []
                for template_token in [f" {node}" for node in nodes] + ["\n", ".", " .", "=", " ="]:
                    template_tokens.append(knowledge_model.tiktoken.encode(template_token)[0])
                if len(logit_bias) < 300:
                    limit_flag = 0
                    for value in values:
                        if limit_flag:
                            break
                        for v in value.values():
                            if limit_flag:
                                break
                            if bool(re.search(r'\d', v)):
                                # do not add value containing numbers
                                continue
                            # if v == "PersonX":
                            if v == f"{rule.variables[node].datatype}X":
                                continue
                            tokens = set(knowledge_model.tiktoken.encode(v))
                            for token in tokens:
                                if len(logit_bias) < 300 and token not in template_tokens:
                                    logit_bias[token] = bias_value
                                else:
                                    # reach the limit
                                    limit_flag = 1
                                    break
                # print(logit_bias)
                # print(f'len: {len(logit_bias)}')
            else:
                current_values = []
                for value in final_values:
                    tmp_str = []
                    for node in value:
                        if node not in generic_nodes:
                            tmp_str.append(f"{node}={value[node]}")
                    tmp_str = ", ".join(tmp_str)
                    current_values.append(tmp_str)
                deduplicate_prompt = f" Do not generate these values: {'; '.join(current_values)}."
                prompt = base_prompt + deduplicate_prompt
                logger.info(f"Prompt:\n{prompt}")
        logger.info(f"Duplicate Count: {duplicate_cnt}")
    else:
        outputs = knowledge_model.generate([prompt], num_sample=call_times, temperature=0.7, max_tokens=512)[0]
        for o in outputs:
            logger.info(f'Generated Text: \n{o}')
            values = knowledge_model.post_process(o.strip(), nodes)
            for v in values:
                for node in generic_nodes:
                    # v[node] = "PersonX"
                    v[node] = f"{rule.variables[node].datatype}X"
                if v not in final_values:
                    final_values.append(v)
    with open(output_file + f"_{distribution}", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(final_values[0].keys()))
        writer.writeheader()
        writer.writerows(final_values)
    return final_values


def load_value_cache(value_cache_file, nodes_to_ignore):
    """Load pre-searched value from file using csv.DictReader"""
    with open(value_cache_file, "r") as f:
        data = pd.read_csv(f, dtype=str)
        data = data.drop(columns=nodes_to_ignore)
        data.drop_duplicates(inplace=True, ignore_index=True)
        final_value_cache = data.to_dict('records')
    return final_value_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--beam_size", type=int, default=40)
    parser.add_argument("--search_n_sample", type=int, default=10)
    parser.add_argument("--output_directory", type=str, default="./output/baseline_output")
    parser.add_argument("--knowledge_model_path", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--rule_keys", nargs="+", default=["rule1", "rule3", "rule4"])
    parser.add_argument("--do_search", action="store_true")
    parser.add_argument("--distribution", nargs="+", default=["head", "longtail"])
    parser.add_argument("--deduplicate", action="store_true", help="Whether or not suppress the likelihood of previous results when searching")
    parser.add_argument("--bias_value", type=int, default=-100)
    parser.add_argument("--use_logit_bias", action="store_true", help="if False, append values to the prompt")
    parser.add_argument("--full", action="store_true", help="generated with full rule or succinct rule")
    parser.add_argument("--meta_rule_info", type=str, default=None)
    
    args = parser.parse_args()

    log_path = f"baseline_{args.knowledge_model_path}_{len(args.rule_keys)}_{args.distribution}_{args.full}_{args.use_logit_bias}.log"
    logging.basicConfig(filename=log_path,
                    filemode='a',format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    logger.info(args)

    if args.do_search:
        knowledge_model = KnowledgeModel(args.knowledge_model_path)

    def name_fn(x):
        if "/" in x:
            x = x.rsplit("/")[1]
        return x
    output_directory = args.output_directory + f"/baseline_{name_fn(args.knowledge_model_path)}"

    for rule_key in args.rule_keys:
        logger.info(f"We are searching for rule {rule_key}")
        rule, graph, _, _, premise_template, conclusion_template = get_rule(rule_key, args.meta_rule_info)

        logger.info("Output dir is {}".format(output_directory))
        pathlib.Path(output_directory).mkdir(exist_ok=True, parents=True)

        if args.do_search:
            value_cache_file = f"{output_directory}/{rule_key}_value_cache.csv"
            for distribution in args.distribution:
                if os.path.exists(value_cache_file + f"_{distribution}"):
                    logger.info(f"{value_cache_file}_{distribution} already exist")
                    continue
                final_values = run_search(knowledge_model, rule, premise_template, conclusion_template, args, value_cache_file, distribution, args.use_logit_bias)
                logger.info("Final values: {}".format(final_values))
        else:
            value_cache_file = f"{output_directory}/{rule_key}_value_cache.csv"
            for distribution in args.distribution:
                final_values = load_value_cache(value_cache_file + f"_{distribution}", [])
                logger.info("Final values: {}".format(final_values))
