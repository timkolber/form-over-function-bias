import json
import statistics
from typing import List, Tuple

import plotly.express as px

from evaluation.analyze_results import map_vote_to_value


def aggregate_pos_bias_voting_data(
    data: dict, better_model_name: str, worse_model_name: str
) -> Tuple:
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

    total = [0, 0, 0]
    for _, answers in question_results.items():
        for answer in answers:
            if answer == 1.0:
                total[0] += 1
            elif answer == 0.0:
                total[1] += 1
            elif answer == 0.5:
                total[2] += 1

    num_parsed_questions = sum(total)
    first_pos_percentage = total[0] / num_parsed_questions
    second_pos_percentage = total[1] / num_parsed_questions
    results = (
        first_pos_percentage,
        second_pos_percentage,
        1.0 - first_pos_percentage - second_pos_percentage,
    )

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
    return (results, final_results)


def calculate_position_bias(file: str):
    with open(file) as f:
        answer_dict = json.load(f)
        results_simple, results = aggregate_pos_bias_voting_data(
            answer_dict, "gpt-4.1_1", "gpt-4.1_2"
        )

        total = [0, 0, 0]
        for question in results.keys():
            if results[question]["winner"] == "gpt-4.1_1":
                total[0] += 1
            elif results[question]["winner"] == "gpt-4.1_2":
                total[1] += 1
            elif results[question]["winner"] == "tie":
                total[2] += 1

        num_parsed_questions = sum(total)
        first_pos_percentage = total[0] / num_parsed_questions
        second_pos_percentage = total[1] / num_parsed_questions

    return (
        results_simple,
        (
            first_pos_percentage,
            second_pos_percentage,
            1.0 - first_pos_percentage - second_pos_percentage,
        ),
    )


def plot_position_bias(
    data: List[Tuple[float, float, float]], models: List[str], file: str
) -> None:

    fig = px.bar(
        x=models,
        y=[[d[0] for d in data], [d[2] for d in data], [d[1] for d in data]],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    # Label the groups
    fig.data[0].name = "First Position"
    fig.data[1].name = "TIE"
    fig.data[2].name = "Second Position"
    # Ensure y-axis label is explicit
    fig.update_yaxes(title_text="Percentage of All Answers")
    fig.update_xaxes(title_text="")
    fig.update_layout(legend_title_text="Chosen Answer")
    # Write to output file
    fig.write_image(file, "svg")


if __name__ == "__main__":

    judge_paths = [
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Llama-3.1-8B-Instruct/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Llama-3.3-70B-Instruct/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Mistral-7B-Instruct-v0.3/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Mistral-Small-3.2-24B-Instruct-2506/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/phi-4/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Qwen3-4B-Instruct-2507/judgements_short_prompt.json",
        "../debug/judgements/gpt-4.1/vs_gpt-4.1/Qwen3-Next-80B-A3B-Instruct/judgements_short_prompt.json",
    ]

    pos_biases = [calculate_position_bias(judge) for judge in judge_paths]
    print(pos_biases)

    judges = [
        "Llama-3.1-8B-Instruct",
        "Llama-3.3-70B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Mistral-Small-3.2-24B-Instruct-2506",
        "phi-4",
        "Qwen3-4B-Instruct-2507",
        "Qwen3-Next-80B-A3B-Instruct",
    ]

    plot_position_bias(
        [pos_bias[0] for pos_bias in pos_biases],
        judges,
        "../outputs/figures/position_biases.svg",
    )
