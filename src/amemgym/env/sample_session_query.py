import json
from amemgym.utils import call_llm
from amemgym.utils.prompt_loader import load_prompts
from loguru import logger


def _normalize_value(value):
    """
    Normalize a value by converting single-element lists to strings.

    Args:
        value: The value to normalize

    Returns:
        The normalized value (string if it was a single-element list)
    """
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value


def sample_update_queries(llm_config, start_date, user_profile, state_schema, updates, lang: str = "en"):
    """
    Sample user queries that can be used to update the user's personal state.

    Args:
        llm_config (dict): Configuration for the LLM.
        start_date (str): The start date in YYYY-MM-DD format.
        user_profile (str): The user's profile containing background information as a formatted string on start_date.
        state_schema (dict): A schema defining the state variables and their possible values.
        updates (dict): A dictionary of state variables to be updated with their new values.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        list: A list of sampled queries that can be used to update the user's personal state.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    old_state = updates["old"]
    new_state = updates["updated"]
    covered = {k: False for k in updates["updated"]}
    context = []
    for event in updates["events"]:
        context_i = {"background": event["event"], "state_transition": {}}
        for k in event["states"]:
            covered[k] = True
            context_i["state_transition"][k] = {
                "old": old_state[k],
                "new": new_state[k]
            }
        context.append(context_i)
    assert all(covered.values()), "Not all state variables are covered in the updates"

    prompt = env_prompts["sample_update_queries_prompt"].format(
        start_date=start_date,
        user_profile_json=json.dumps(user_profile, indent=2, ensure_ascii=False),
        period_start=updates["period_start"],
        period_end=updates["period_end"],
        context_json=json.dumps(context, indent=2, ensure_ascii=False),
        state_schema_json=json.dumps(state_schema, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)

    try:
        queries = json.loads(response)["queries"]
        if len(queries) != len(updates["events"]):
            raise ValueError(f"Number of queries {len(queries)} does not match number of updates {len(updates['events'])}")
        outputs = []
        for query, event in zip(queries, updates["events"]):
            # Normalize query format
            query = _normalize_value(query)
            exposed_states = {k: new_state[k] for k in event["states"]}
            outputs.append({
                "query": query,
                "exposed_states": exposed_states
            })
        return outputs
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        raise e


def sample_init_queries(llm_config, start_date, user_profile, state_schema, initial_state, lang: str = "en"):
    """
    Sample several queries that can be used to expose the user's initial state.

    Args:
        llm_config (dict): Configuration for the LLM.
        start_date (str): The start date in YYYY-MM-DD format.
        user_profile (str): The user's profile containing background information as a formatted string on start_date.
        state_schema (dict): A schema defining the state variables and their possible values.
        initial_state (dict): The current initial state of the user's personal variables.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        list: A list of sampled queries that can be used to expose the user's personal state.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    prompt = env_prompts["sample_init_queries_prompt"].format(
        start_date=start_date,
        user_profile=user_profile,
        initial_state_json=json.dumps(initial_state, indent=2, ensure_ascii=False),
        state_schema_json=json.dumps(state_schema, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    try:
        queries = json.loads(response)["queries"]
        exposed_states = set()
        for state in queries:
            # Normalize query format
            state["query"] = _normalize_value(state.get("query", ""))
            for key, value in state["exposed_states"].items():
                # Normalize exposed state values
                value = _normalize_value(value)
                assert value == initial_state[key], f"Exposed state {key} has unexpected value {value}, expected {initial_state[key]}"
                exposed_states.add(key)
        assert len(exposed_states) == len(initial_state), "Not all initial states were exposed in the queries"
        return queries
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        raise e


def check_query_state_exposure(llm_config, query, exposed_states, state_schema, lang: str = "en"):
    """
    Check if a query correctly exposes the intended state variables.

    Args:
        llm_config (dict): Configuration for the LLM.
        query (str): The user's query.
        exposed_states (dict): A dictionary mapping state variable names to their intended values.
        state_schema (dict): A schema defining the state variables and their possible values.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        bool: True if the query correctly exposes the intended states, False otherwise.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    state_choices = {k: state_schema[k] for k in exposed_states}

    prompt = env_prompts["check_query_state_exposure_prompt"].format(
        query=query,
        state_choices_json=json.dumps(state_choices, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    try:
        predicted_states = json.loads(response)
        for state_var, expected_value in exposed_states.items():
            if state_var not in predicted_states:
                logger.warning(f"State variable '{state_var}' not predicted")
                return False

            # Normalize format: convert list to string if needed
            predicted_value = _normalize_value(predicted_states[state_var])

            if predicted_value != expected_value:
                logger.warning(f"State variable '{state_var}': predicted '{predicted_value}', expected '{expected_value}'")
                return False
        return True

    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        return False


def refine_query(llm_config, query, exposed_states, state_schema, lang: str = "en"):
    """
    Refine a user query to better expose the intended state variables.

    Args:
        llm_config (dict): Configuration for the LLM.
        query (str): The original user query.
        exposed_states (dict): A dictionary mapping state variable names to their intended values.
        state_schema (dict): A schema defining the state variables and their possible values.
        lang (str): Language code for prompt selection (default: "en").

    Returns:
        str: The refined query that better exposes the intended states.
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)

    state_choices = {k: state_schema[k] for k in exposed_states}

    prompt = env_prompts["refine_query_prompt"].format(
        query=query,
        exposed_states_json=json.dumps(exposed_states, indent=2, ensure_ascii=False),
        state_choices_json=json.dumps(state_choices, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)

    try:
        refined_query = json.loads(response)["query"]
        refined_query = _normalize_value(refined_query)
        return refined_query.strip()
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
        logger.error("Response content:", response)
        raise e
