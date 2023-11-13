import os
import openai
from collections import defaultdict
import csv
import random
random.seed(42)
import json
import argparse
import logging
import numpy as np
import pandas as pd
import pathlib
import evaluate
from knowledge_beam_search import OpenAIWrapper, _swap_conclusion_to_last, ProbingModel, get_rule, load_value_cache, _get_full_rule_predicates
rouge = evaluate.load("rouge")
os.environ["TIKTOKEN_CACHE_DIR"] = ""
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY") 


def diverse_prompt_format(premise, conclusion, collect_rationale=False, cot=False, positive_conclusion=False):

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
    if not positive_conclusion:
        # conclusion is: xx cannot do xx
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
    else:
        # conclusion is: xx can do xx
        if not cot:
            if collect_rationale:
                positive_label_formats = [
                    (f"Is it true that if {premise}, {conclusion[:-1]}? Answer yes or no, and then give an explanation to your answer:", "Yes"),
                    (f"Yes or no: if {premise}, {conclusion} Also give an explanation to the answer.", "Yes"),
                    (f"Answer the question with yes or no, and then give an explanation to your answer: if {premise}, {conclusion_question}", "Yes"),
                    (f"True or false: if {premise}, {conclusion} Also give an explanation to the answer.", "True"),
                    (f"Right or Wrong: if {premise}, {conclusion} Also give an explanation to the answer.", "Right"),
                    (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion? Answer yes or no, and then give an explanation to your answer:", "Yes"),
                    (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion? Answer yes or no, and then give an explanation to your answer:", "Yes"),
                ]
                negative_label_formats = [
                    (f"Is it true that if {premise}, {conclusion_negation[:-1]}? Answer yes or no, and then give an explanation to your answer:", "No"),
                    (f"Yes or no: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "No"),
                    (f"True or false: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "False"),
                    (f"Right or Wrong: if {premise}, {conclusion_negation} Also give an explanation to the answer.", "Wrong"),
                    (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion? Answer yes or no, and then give an explanation to your answer:", "No"),
                    (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion? Answer yes or no, and then give an explanation to your answer:", "No"),
                ]
            else:
                positive_label_formats = [
                    (f"Is it true that if {premise}, {conclusion[:-1]}? Answer yes or no:", "Yes"),
                    (f"Yes or no: if {premise}, {conclusion}", "Yes"),
                    (f"Answer the question with yes or no: if {premise}, {conclusion_question}", "Yes"),
                    (f"True or false: if {premise}, {conclusion}", "True"),
                    (f"Right or Wrong: if {premise}, {conclusion}", "Right"),
                    (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion? Answer yes or no:", "Yes"),
                    (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion? Answer yes or no:", "Yes"),
                ]
                negative_label_formats = [
                    (f"Is it true that if {premise}, {conclusion_negation[:-1]}? Answer yes or no:", "No"),
                    (f"Yes or no: if {premise}, {conclusion_negation}", "No"),
                    (f"True or false: if {premise}, {conclusion_negation}", "False"),
                    (f"Right or Wrong: if {premise}, {conclusion_negation}", "Wrong"),
                    (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion? Answer yes or no:", "No"),
                    (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion? Answer yes or no:", "No"),
                ]
        else:
            positive_label_formats = [
                (f"Is it true that if {premise}, {conclusion[:-1]}?", "Therefore, the answer (yes or no) is:", "Yes"),
                (f"Yes or no: if {premise}, {conclusion}", "Therefore, the answer (yes or no) is:", "Yes"),
                (f"Answer the question: if {premise}, {conclusion_question}", "Therefore, the answer (yes or no) is:", "Yes"),
                (f"True or false: if {premise}, {conclusion}", "Therefore, the answer (true or false) is:", "True"),
                (f"Right or Wrong: if {premise}, {conclusion}", "Therefore, the answer (right or wrong) is:", "Right"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise entail conclusion?", "Therefore, the answer (yes or no) is:", "Yes"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise contradict the conclusion?", "Therefore, the answer (yes or no) is:", "Yes"),
            ]
            negative_label_formats = [
                (f"Is it true that if {premise}, {conclusion_negation[:-1]}?", "Therefore, the answer (yes or no) is:", "No"),
                (f"Yes or no: if {premise}, {conclusion_negation}", "Therefore, the answer (yes or no) is:", "No"),
                (f"True or false: if {premise}, {conclusion_negation}", "Therefore, the answer (true or false) is:", "False"),
                (f"Right or Wrong: if {premise}, {conclusion_negation}", "Therefore, the answer (right or wrong) is:", "Wrong"),
                (f"Premise: {premise}. Conclusion: {conclusion_negation} Does premise entail conclusion?", "Therefore, the answer (yes or no) is:", "No"),
                (f"Premise: {premise}. Conclusion: {conclusion} Does premise contradict the conclusion?", "Therefore, the answer (yes or no) is:", "No"),
            ]
     
    positive_label_formats = [(i[0].replace("if if ","if "),*i[1:]) for i in positive_label_formats]
    negative_label_formats = [(i[0].replace("if if ","if "),*i[1:]) for i in negative_label_formats]

    return range(len(positive_label_formats) + len(negative_label_formats)), positive_label_formats + negative_label_formats


def run_probing(probe_model, final_beams, premise_template, conclusion_template, _get_full_rule_predicates, traverse_order, traverse_order_entity, rule_for_fullrules, output_file, rule_key, rule_probe_type, collect_rationale=False, cot=False, cot_icl = False, positive_conclusion=False):
    """Create probing inputs using values in final beams and on diverse prompt formats, and writes the results to output_file"""
    SPECIAL_INSTRUCTIONS = "For the following question, please answer in a normal life scenario with no special consideration:\n"
    COT_INSTRUCTIONS = "\nLet's think step by step.\n"
    COT_ICL = {"premise_icl":["PersonX was born in Roman Republic","PersonX is allergic to mango"],
               "conclusion_icl":["PersonX was not able to use tractor","PersonX is not able to eat mango rice pudding"],
               "reason_icl":["The Roman Republic existed around 509-27 BC. Tractor was invented in 1892. The Roman Republic was long before when tractor was invented.",
                             "Mango is an ingredient in mango rice pudding."]}
    if cot_icl:
        assert cot, "if cot_icl is true, then must set cot to true"
        all_prompts_cot_icl_cache = {}
        for i in range(len(COT_ICL["premise_icl"])):
            premise = COT_ICL["premise_icl"][i]
            conclusion = COT_ICL["conclusion_icl"][i]
            _, all_prompts_cot_icl = diverse_prompt_format(premise,conclusion, collect_rationale, cot)
            all_prompts_cot_icl_cache[i] = all_prompts_cot_icl

    indices = []
    inputs = []
    cot_answer_inputs = []
    labels = []
    # We want to form a (final_beam_size, num_prompt_format, 1-sentence) array to easily for icl example
    if output_file.endswith("longtail"):
        tail_or_head = "tail"
    else:
        tail_or_head = "head"
    
    for final_beams_indice,value_dict in enumerate(final_beams):
        premise = premise_template
        conclusion = conclusion_template
        
        if rule_probe_type == "succint":
            for key in value_dict:
                print("key: ", key, "value: ", value_dict[key])
                premise = premise.replace("["+key+"]", " ".join(value_dict[key].split("_")))
                conclusion = conclusion.replace("["+key+"]", " ".join(value_dict[key].split("_")))

        # full rules
        if rule_probe_type == "full":
            full_rull_predicates = _get_full_rule_predicates(traverse_order, rule_for_fullrules,graph, traverse_order_entity)
            full_rull_predicates = _swap_conclusion_to_last(full_rull_predicates,list(rule.conclusion_map.keys())[0])
            full_rule_instanitiation = rule_for_fullrules.verbalize_with_partial_rules(full_rull_predicates,value_dict,list(value_dict.keys()))
            premise ,conclusion = full_rule_instanitiation.rsplit("and",1)
            premise, conclusion = premise.strip(), conclusion.strip()

        prompt_indices, all_prompts = diverse_prompt_format(premise, conclusion, collect_rationale, cot, positive_conclusion)
        indices += [f"{rule_key}_{tail_or_head}_{final_beams_indice}_prompt_{prompt_indice}" for prompt_indice in prompt_indices]


        
        if cot:
            if cot_icl:
                inputs += [ prompt[0] + COT_INSTRUCTIONS for prompt in all_prompts]
            else:
                inputs += [ SPECIAL_INSTRUCTIONS + prompt[0] + COT_INSTRUCTIONS for prompt in all_prompts]
            cot_answer_inputs += [prompt[1] for prompt in all_prompts]
        else:
            inputs += [ SPECIAL_INSTRUCTIONS + prompt[0] for prompt in all_prompts]
        labels += [prompt[-1] for prompt in all_prompts]
        
        # indices.append(prompt_indices)
        # inputs.append([prompt[0] for prompt in all_prompts])
        # labels.append([prompt[1] for prompt in all_prompts])
    # inputs_with_icl = form_icl_probing(inputs, labels, indices)
    if cot_icl:
        for i in range(len(inputs)):
            cot_icl_text = ""
            pos_neg_flag = random.sample([0,1], 1)[0] # 0 positive; 1 negative
            for key in all_prompts_cot_icl_cache:
                if pos_neg_flag:
                    # negative example
                    random_index = random.sample(prompt_indices[len(prompt_indices)//2:], 1)[0]
                else:
                    # postive example
                    random_index = random.sample(prompt_indices[:len(prompt_indices)//2], 1)[0]
                tmp = all_prompts_cot_icl_cache[key][random_index][0] + COT_INSTRUCTIONS + COT_ICL["reason_icl"][key] + "\n" + all_prompts_cot_icl_cache[key][random_index][1] + " " + all_prompts_cot_icl_cache[key][random_index][2] + "\n"
                cot_icl_text += tmp
                # change the polarity
                pos_neg_flag = 1 - pos_neg_flag

            cot_icl_text = SPECIAL_INSTRUCTIONS + cot_icl_text
            inputs[i] = cot_icl_text + inputs[i]

    correct, wrong, accuracy = probe_model.probe(inputs, cot_answer_inputs, labels, indices, collect_rationale, cot, cot_icl)
    pathlib.Path(output_file).parent.mkdir(exist_ok= True, parents= True)
    with open(output_file, "w") as f:
        json.dump({"correct": correct, "wrong": wrong, "accuracy": accuracy}, f, indent=4)
    return correct, wrong, accuracy

def summarize_prompt_performance(correct, wrong, filename):
    """Summarize the performance of each prompt and write to file."""
    template_performance = defaultdict(dict)
    for c_index, c_input, *_ in correct:
        if "correct" not in template_performance[c_index]:
            template_performance[c_index]["correct"] = []
        if "wrong" not in template_performance[c_index]:
            template_performance[c_index]["wrong"] = []
        template_performance[c_index]["correct"].append(c_input)
    for w_index, w_input, *_ in wrong:
        if "correct" not in template_performance[w_index]:
            template_performance[w_index]["correct"] = []
        if "wrong" not in template_performance[w_index]:
            template_performance[w_index]["wrong"] = []
        template_performance[w_index]["wrong"].append(w_input)
    index_performance = list(template_performance.items())
    index_performance = sorted(index_performance, key=lambda x: x[0])
    pathlib.Path(filename).parent.mkdir(exist_ok= True, parents= True)
    with open(filename, "w") as writer:
        csv_writer = csv.writer(writer)
        csv_writer.writerow(["Template", "Accuracy"])
        for index, performance in index_performance:
            csv_writer.writerow([index, len(performance["correct"]) / (len(performance["correct"]) + len(performance["wrong"]))])
            # writer.write("Template: " + str(index) + " ")
            # writer.write("Accuracy: " + str(len(performance["correct"]) / (len(performance["correct"]) + len(performance["wrong"]))) + "\n")

def recalculate_acc(path, flip_label_data):
    """
    recalculate the accuracy since there are some beams labeled as incorrect in human eval, their labels should be flipped
    """
    with open(path, "r") as f:
        data = pd.read_csv(f)
        id_l = list(data["Template"])
        acc_l = list(data["Accuracy"])
        neg_correct_cnt = 0
        pos_correct_cnt = 0
        neg_all_cnt = 0
        pos_all_cnt = 0
        # logger.info(flip_label_data)
        for id, acc in zip(id_l, acc_l):
            rule, distribution, beam_id, _, template_id = id.split("_")
            if distribution == "tail":
                distribution = "longtail"
            if int(beam_id) in flip_label_data:
                neg_correct_cnt += (1-acc)
                neg_all_cnt += 1
            else:
                pos_correct_cnt += acc
                pos_all_cnt += 1
        if neg_all_cnt + pos_all_cnt != 0:
            acc = (neg_correct_cnt + pos_correct_cnt) / (neg_all_cnt + pos_all_cnt)
        else:
            acc = None
        if neg_all_cnt != 0:
            hard_acc = neg_correct_cnt / neg_all_cnt
        else:
            hard_acc = None
        if pos_all_cnt != 0:
            easy_acc = pos_correct_cnt / pos_all_cnt
        else:
            easy_acc = None
        if len(acc_l) != 0:
            previous_acc = np.sum(acc_l) / len(acc_l)
        else:
            previous_acc = None
        logger.info(f"Overall acc: {acc}")
        logger.info(f"ACC of hard samples: {hard_acc}, {neg_correct_cnt}, {neg_all_cnt}")
        logger.info(f"ACC of easy samples: {easy_acc}, {pos_correct_cnt}, {pos_all_cnt}"),
        logger.info(f"Previous acc: {previous_acc}")
        return neg_correct_cnt + pos_correct_cnt, neg_all_cnt + pos_all_cnt, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str, default="to_be_replaced")
    parser.add_argument("--probe_model_path", type=str, default="text-davinci-003")
    parser.add_argument("--probe_tokenizer_path", type=str, default=None)
    parser.add_argument("--rule_keys", nargs="+", default=["rule1", "rule3", "rule4"])
    parser.add_argument("--do_probe", action="store_true") 
    parser.add_argument("--method_name", type=str, default="new_rules_v4")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--cot_icl", action="store_true")
    parser.add_argument("--collect_rationale", action="store_true", help="whether or not collect rationale when probing")
    parser.add_argument("--rule_path", type=str, default=None)
    parser.add_argument("--positive_conclusion_rules", nargs="+", default=[])
    parser.add_argument("--traverse_order", type=str, default="conclusion", help="Whether to traverse the graph in conclusion-first or premise-first order", choices=["conclusion", "premise"])
    parser.add_argument("--rule_probe_type",choices=["succint","full"],default="succint")
    
    args = parser.parse_args()
    log_path = f"../log/new_algo_probing/knowledgesearch_probe_{args.probe_model_path}_{len(args.rule_keys)}_{args.cot_icl}.log"
    logging.basicConfig(filename=log_path,
                    filemode='a',format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
    logger.info(args)
    logger.info(openai.api_key)

    if args.do_probe:
        probe_model = ProbingModel(args.probe_model_path)
    
    output_directory = args.output_directory

    # there should be flip_label_data.json in the output_directory
    with open(f"{output_directory}/flip_label_data.json", "r") as f:
        flip_label_data = json.load(f)[args.method_name]

    correct_cnt_head = 0
    correct_cnt_longtail = 0
    all_cnt_head = 0
    all_cnt_longtail = 0

    # rule of interest
    for rule_key in args.rule_keys:
        traverse_order_force_conclusion = False
        logger.info(f"We are probing for rule {rule_key}")
        rule, graph, conclusion_first_order, premise_first_order, premise_template, conclusion_template = get_rule(rule_key, args.rule_path)

        traverse_order_entity = "conclusion" if args.traverse_order == "conclusion" or traverse_order_force_conclusion else "premise"
        # force the rules which have factual node to be conclusion first, otherwise premise first
        traverse_order = conclusion_first_order if traverse_order_entity == "conclusion" else premise_first_order
        
        value_cache_file = f"{output_directory}/{rule_key}_value_cache.csv"
        head_final_beams, longtail_final_beams = load_value_cache(value_cache_file, [])

        if args.do_probe:
            
            probe_result_file = f"{output_directory}/{rule_key}_{args.probe_model_path}_cot_{args.cot}_cot_icl_{args.cot_icl}_probe_results_{args.rule_probe_type}.json"
            template_result_file = f"{output_directory}/{rule_key}_{args.probe_model_path}_cot_{args.cot}_cot_icl_{args.cot_icl}_performance_by_template_{args.rule_probe_type}.csv"

            if rule_key in args.positive_conclusion_rules:
                positive_conclusion = True
            else:
                positive_conclusion = False
            
            if not (os.path.exists(probe_result_file + "_longtail") and os.path.exists(template_result_file + "_longtail")):
                l_correct, l_wrong, l_accuracy = run_probing(probe_model, longtail_final_beams, premise_template, conclusion_template, _get_full_rule_predicates, traverse_order, traverse_order_entity, rule, probe_result_file + "_longtail",rule_key, args.rule_probe_type, args.collect_rationale, args.cot, args.cot_icl, )
                h_correct, h_wrong, h_accuracy = run_probing(probe_model, head_final_beams, premise_template, conclusion_template, _get_full_rule_predicates, traverse_order, traverse_order_entity, rule, probe_result_file + "_head",rule_key, args.rule_probe_type, args.collect_rationale, args.cot, args.cot_icl)
                
                summarize_prompt_performance(l_correct, l_wrong, template_result_file + "_longtail")
                summarize_prompt_performance(h_correct, h_wrong, template_result_file + "_head")

            l_correct, l_all, l_acc = recalculate_acc(template_result_file + "_longtail", flip_label_data[rule_key]["longtail"])
            h_correct, h_all, h_acc = recalculate_acc(template_result_file + "_head", flip_label_data[rule_key]["head"])
            all_cnt_head += h_all
            all_cnt_longtail += l_all
            correct_cnt_head += h_correct
            correct_cnt_longtail += l_correct
    logger.info(f"ACC of all head beams: {correct_cnt_head / all_cnt_head}")
    logger.info(f"ACC of all longtail beams: {correct_cnt_longtail / all_cnt_longtail}")
    aggregate_acc_path = f"{output_directory}/aggregate_{args.probe_model_path}_cot_{args.cot}_cot_icl_{args.cot_icl}_probe_results_{args.rule_probe_type}.json"
    aggregate_acc = {
        "head": {
            "correct": correct_cnt_head,
            "all": all_cnt_head,
            "accuracy": correct_cnt_head / all_cnt_head
        },
        "longtail": {
            "correct": correct_cnt_longtail,
            "all": all_cnt_longtail,
            "accuracy": correct_cnt_longtail / all_cnt_longtail
        }
    }
    with open(aggregate_acc_path, "w") as f:
        json.dump(aggregate_acc, f, indent=2)