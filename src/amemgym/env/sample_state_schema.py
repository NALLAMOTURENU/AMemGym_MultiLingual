import json
from amemgym.utils import call_llm
from amemgym.utils.prompt_loader import load_prompts
from loguru import logger


def sample_user_questions(
    llm_config, start_date, user_profile, num_questions=10, num_states_per_question=2, num_choices_per_state=3, num_total_months=30, lang="en"
):
    """
    Samples a set of potential questions from the user with the given user profile.
    The questions are designed to be asked by the user for suggestions or advice,
    and require specific personal information to answer. They can be asked at any time in
    several years, regardless of the user's development and experience at that time.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        start_date (datetime): start date as a datetime object
        user_profile (str): User profile as a formatted string
        num_questions (int): Number of questions to generate (default: 10)
        num_states_per_question (int): Number of required_info items per question (default: 2)
        num_choices_per_state (int): Number of choices per required_info item (default: 3)
        num_total_months (int): Total months to consider for user development (default: 30)
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        list[dict]: List of question dictionaries with required_info
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)
    detailed_prompt = env_prompts["sample_user_questions_prompt"].format(
        start_date=start_date,
        user_profile=user_profile,
        num_questions=num_questions,
        num_total_months=num_total_months,
        num_states_per_question=num_states_per_question,
        num_choices_per_state=num_choices_per_state,
    )

    response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)

    # Handle response parsing
    try:
        questions = json.loads(response)["questions"]
        return questions
    except Exception as e:
        logger.error(f"Failed to parse questions response: {e}\nRaw response: {response}")
        return []


def refine_state_schema(llm_config, user_profile, questions, lang="en"):
    """
    Refines the user profile schema based on the sampled questions.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        user_profile (str): user profile as a formatted string
        questions (list[dict]): List of question dictionaries with required_info
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Specific type of information needed
                - info_choices (list[str]): Choices for this info type
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        dict: Refined persona schema with exclusive info types mapped to original info types.
            - key (str): new exclusive and unambiguous info type
            - value (list[str]): original info types that map to this key
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)
    detailed_prompt = env_prompts["refine_state_schema_prompt"].format(
        user_profile=user_profile,
        questions_json=json.dumps(questions, ensure_ascii=False, indent=2),
    )

    response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)

    # Handle response parsing
    try:
        refined_schema = json.loads(response)
        # ensure all info_types in questions are covered in the refined schema
        orig2new = {}
        for new_type, orig_types in refined_schema.items():
            for orig_type in orig_types:
                orig2new[orig_type] = new_type
        for question in questions:
            for info in question["required_info"]:
                info_type = info["info_type"]
                assert info_type in orig2new, f"Info type '{info_type}' not found in refined schema"
        return refined_schema
    except Exception as e:
        logger.error(f"Failed to parse refined schema response: {e}\nRaw response: {response}")
        return {}


def fix_schema_inconsistencies(
    llm_config, start_date, user_profile, num_total_months,
    num_choices_per_state, questions, refined_schema, lang="en"
):
    """
    Fix inconsistencies in the user profile based on the refined schema.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        start_date (str): Start date in format "YYYY-MM-DD"
        user_profile (str): User profile as a formatted string
        num_total_months (int): Total months to consider for user development
        num_choices_per_state (int): Number of choices per required_info item
        questions (list[dict]): List of original question dictionaries with required_info
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Specific type of information needed
                - info_choices (list[str]): Choices for this info type
        refined_schema (dict): Refined persona schema with exclusive info types
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        dict: Updated questions with updated info_types and choices based on the refined schema.
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Updated specific type of information needed
                - info_choices (list[str]): Updated choices for this info type
        dict: Updated state schema with new exclusive info types and their choices.
            - key (str): new exclusive and unambiguous info type
            - value (list[str]): choices for this info type
    """
    # update questions based on refined schema and check possible inconsistencies
    origtype2choices = {}
    for question in questions:
        q = question["question"]
        for info in question["required_info"]:
            info_type = info["info_type"]
            if info_type not in origtype2choices:
                origtype2choices[info_type] = []
            origtype2choices[info_type].append([q, info["info_choices"]])

    state_schema, orig2newtype = {}, {}
    conflict_groups = {}
    for new_type, orig_types in refined_schema.items():
        choices = []
        for orig_type in orig_types:
            choices.extend(origtype2choices[orig_type])
            orig2newtype[orig_type] = new_type
        if len(choices) == 1:
            state_schema[new_type] = choices[0][1]
        else:
            conflict_groups[new_type] = [
                {"question": item[0], "choices": item[1]} for item in choices
            ]

    # Resolve all conflicts in a single LLM call
    newtype2choices = {}

    if conflict_groups:
        env_prompts = load_prompts("env", lang=lang, escape=False)
        detailed_prompt = env_prompts["fix_schema_inconsistencies_prompt"].format(
            start_date=start_date,
            user_profile=user_profile,
            conflict_groups_json=json.dumps(conflict_groups, ensure_ascii=False, indent=2),
            num_choices_per_state=num_choices_per_state,
            num_total_months=num_total_months,
        )

        response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)

        try:
            newtype2choices = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse conflict resolution response: {e}\nRaw response: {response}")
            raise ValueError("Failed to parse conflict resolution response.")

    # Validate that all conflict types have been resolved
    for new_type, q_choices in conflict_groups.items():
        if new_type not in newtype2choices:
            logger.error(f"Failed to get new choices for type '{new_type}' from batch resolution")
            raise ValueError(f"Failed to resolve conflict for type '{new_type}'.")
        state_schema[new_type] = newtype2choices[new_type]

    # update questions with new choices
    updated_questions = []
    for question in questions:
        new_question = {
            "question": question["question"],
            "required_info": []
        }
        for info in question["required_info"]:
            info_type = info["info_type"]
            new_type = orig2newtype[info_type]
            new_question["required_info"].append({
                "info_type": new_type,
                "info_choices": state_schema[new_type]
            })
        updated_questions.append(new_question)

    return updated_questions, state_schema


if __name__ == "__main__":
    pass
