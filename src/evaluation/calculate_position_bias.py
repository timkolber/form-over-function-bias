import json
from analyze_results import map_vote_to_value
from typing import Dict, Union
import statistics


def aggregate_pos_bias_voting_data(
    data: dict, better_model_name: str, worse_model_name: str, without_majority_vote: bool=False
) -> Union[Dict[str, Dict[str, float]], float]:
    """Extract and process question results for the given model pair."""
    question_results = {}
    n = 0
    for question_group_key in data.keys():
        if question_group_key == "metadata":
            continue

        question_group = data[question_group_key]

        for question_data in question_group:
            extracted_answers = question_data["extracted_answers"]
            extracted_answers = [ans for ans in extracted_answers]

            question_results[n] = []

            for vote in extracted_answers:
                vote_value = map_vote_to_value(
                    vote, better_model_name, worse_model_name
                )
                if vote_value is not None:
                    question_results[n].append(vote_value)

            n += 1

    if without_majority_vote:
        total = [0,0,0]
        for _ , answers in question_results.items():
            for answer in answers:
                if answer == 1.0:
                    total[0] += 1
                elif answer == 0.0:
                    total[1] += 1
                elif answer == 0.5:
                    total[2] += 1
        
        print(total)
        num_parsed_questions = sum(total)
        print(num_parsed_questions)
        first_pos_percentage = total[0] / num_parsed_questions
        print(first_pos_percentage)
        second_pos_percentage = total[1] / num_parsed_questions
        print(second_pos_percentage)
    return first_pos_percentage - second_pos_percentage

    final_results = {}
    for n, all_votes in question_results.items():
        if all_votes:
            better_model_avg = statistics.mean(all_votes)
            final_results[n] = {
                f"{better_model_name}_avg": better_model_avg,
                f"{worse_model_name}_avg": 1.0 - better_model_avg,
                "winner": (
                    better_model_name
                    if better_model_avg > 0.5
                    else worse_model_name if better_model_avg < 0.5 else "tie"
                ),
                "total_votes": len(all_votes),
            }
    return final_results

def calculate_position_bias(file:str):
    with open(file) as f:
        answer_dict = json.load(f)
        results = aggregate_pos_bias_voting_data(answer_dict, "gpt-4.1_1", "gpt-4.1_2", without_majority_vote=True)
        return results
    #     print(results)
    #     total = [0,0,0]
    #     for question in results.keys():
    #         if results[question]["winner"] == "gpt-4.1_1":
    #             total[0] += 1
    #         elif results[question]["winner"] == "gpt-4.1_2":
    #             total[1] += 1
    #         elif results[question]["winner"] == "tie":
    #             total[2] += 1
        
    #     print(total)
    #     num_parsed_questions = sum(total)
    #     print(num_parsed_questions)
    #     first_pos_percentage = total[0] / num_parsed_questions
    #     print(first_pos_percentage)
    #     second_pos_percentage = total[1] / num_parsed_questions
    #     print(second_pos_percentage)
    # return first_pos_percentage - second_pos_percentage


models = ["./debug/judgements/gpt-4.1/vs_gpt-4.1/Llama-3.1-8B-Instruct/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/Llama-3.3-70B-Instruct/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/Mistral-7B-Instruct-v0.3/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/Mistral-Small-3.2-24B-Instruct-2506/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/phi-4/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/Qwen3-4B-Instruct-2507/judgements_short_prompt.json", 
          "./debug/judgements/gpt-4.1/vs_gpt-4.1/Qwen3-Next-80B-A3B-Instruct/judgements_short_prompt.json"]

print([calculate_position_bias(model) for model in models])