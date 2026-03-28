import json
import itertools
import random

from amemgym.utils.llm_utils import call_llm
from amemgym.utils.prompt_loader import load_prompts
from loguru import logger


def sample_personalized_answers(llm_config, question, state_variants, lang: str = "en"):
    """
    Sample personalized answers for a question based on state variants.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
            - 'question' (str): The question text.
            - 'required_info' (list): List of required information items, each with 'info_type' and 'info_choices'.
                - 'info_type' (str): The type of information required.
                - 'info_choices' (list): List of choices for this information type.
        state_variants (list): List of state variants to consider.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        list: A list of dictionaries, each containing:
            - 'variant': The state variant as a list of values (for json compatibility).
            - 'answer': The personalized answer for that variant.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    required_info_types = [info['info_type'] for info in question['required_info']]

    variants_text = ""
    for i, variant in enumerate(state_variants, 1):
        variant_info = []
        for info_type, choice in zip(required_info_types, variant):
            variant_info.append(f"{info_type}: {choice}")
        variants_text += f"Variant {i}: {', '.join(variant_info)}\n"

    prompt = env_prompts["sample_personalized_answers_prompt"].format(
        question=question['question'],
        required_info_types=', '.join(required_info_types),
        variants_text=variants_text,
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)

    try:
        parsed_response = json.loads(response)

        answers = []
        for i, variant in enumerate(state_variants, 1):
            variant_key = f"variant_{i}"
            if variant_key not in parsed_response:
                logger.warning(f"Missing answer for {variant_key}")
            answers.append({
                "variant": list(variant),
                "answer": parsed_response[variant_key]
            })
        return answers

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing personalized answers: {e}\nRaw response: {response}")
        raise e


def check_personalized_answer(llm_config, question, answer, variants, matched_index, lang: str = "en"):
    """
    Validate a personalized answer to ensure it aligns with the question and required information.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
        answer (str): The personalized answer to validate.
        variants (list): List of state variants considered.
        matched_index (int): The index of the variant that this answer is supposed to correspond to.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        bool: True if the answer matches the expected variant.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    required_info_types = [info['info_type'] for info in question['required_info']]
    choices = "\n".join([
        f"{i+1}. " + json.dumps({k: v for k, v in zip(required_info_types, variant)})
        for i, variant in enumerate(variants)
    ])

    prompt = env_prompts["check_personalized_answer_prompt"].format(
        question=question['question'],
        answer=answer,
        choices=choices,
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=False)
    try:
        choice_index = int(response.strip()) - 1
        if choice_index == matched_index:
            return True
    except (ValueError, TypeError):
        pass
    return False


def refine_personalized_answer(llm_config, question, answer, variants, matched_index, lang: str = "en"):
    """
    Refine a personalized answer to better align with the question and required information.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
        answer (str): The personalized answer to refine.
        variants (list): List of state variants considered.
        matched_index (int): The index of the variant that this answer is supposed to correspond to.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        str: The refined answer.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    required_info_types = [info['info_type'] for info in question['required_info']]
    matched_state = json.dumps({k: v for k, v in zip(required_info_types, variants[matched_index])})
    other_states = [
        {k: v for k, v in zip(required_info_types, variant)}
        for i, variant in enumerate(variants) if i != matched_index
    ]
    other_states_text = "\n".join([json.dumps(state) for state in other_states])

    prompt = env_prompts["refine_personalized_answer_prompt"].format(
        question=question['question'],
        matched_state=matched_state,
        other_states_text=other_states_text,
        answer=answer,
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    try:
        parsed_response = json.loads(response)
        return parsed_response.get("answer", answer)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing refined answer: {e}\nRaw response: {response}")
        return answer


def get_state_variants(states, questions, min_variants=4):
    """
    Generate a list of state variants based on the evolution of state variables and the questions asked.

    Args:
        states (list): A list of state snapshots, where each snapshot is a dictionary of state variables and their values.
        questions (list): A list of questions to be asked.
        min_variants (int): Minimum number of state variants to ensure diversity in answers.

    Returns:
        dict: A dictionary mapping each question to a list of unique state variants
    """
    state_variants = {}
    for question in questions:
        info_types = [info["info_type"] for info in question["required_info"]]
        variants = set()
        for state in states:
            variant = tuple(state[info_type] for info_type in info_types)
            variants.add(variant)

        variant_list = list(variants)

        if len(variant_list) < min_variants:
            info_choices_lists = [info["info_choices"] for info in question["required_info"]]
            all_combinations = list(itertools.product(*info_choices_lists))
            remaining_combinations = list(set(all_combinations) - variants)
            additional_needed = min_variants - len(variant_list)
            assert len(remaining_combinations) >= additional_needed, \
                "Not enough unique combinations available to fit the minimum variants requirement"
            additional_variants = random.sample(remaining_combinations, additional_needed)
            variant_list.extend(additional_variants)

        state_variants[question["question"]] = variant_list
    return state_variants
