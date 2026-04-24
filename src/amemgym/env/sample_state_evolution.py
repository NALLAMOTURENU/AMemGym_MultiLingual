import json
from amemgym.utils import call_llm, find_best_semantic_match
from amemgym.utils.prompt_loader import load_prompts
from loguru import logger
from backoff import on_exception, expo


def sample_initial_state(llm_config, start_date, user_profile, num_total_months, state_schema, lang="en"):
    """
    Sample initial state values for a user's personal state variables based on their current profile and a predefined schema.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc
        start_date (str): The current date in YYYY-MM-DD format.
        user_profile (dict): The user's profile containing personal information.
        num_total_months (int): The total number of months to consider for future state evolution.
        state_schema (dict): A schema defining the state variables and their possible values.
        lang (str): Language code for prompt selection (default: "en")
    """
    env_prompts = load_prompts("env", lang=lang, escape=False)
    prompt = env_prompts["sample_initial_state_prompt"].format(
        num_total_months=num_total_months,
        start_date=start_date,
        user_profile=user_profile,
        state_schema_json=json.dumps(state_schema, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)

    try:
        initial_state = json.loads(response)

        # Validate that all keys exist in schema and values are valid choices
        for key, value in initial_state.items():
            if key not in state_schema:
                raise ValueError(f"Invalid state variable: {key}")

            # Semantic matching
            best_match, score = find_best_semantic_match(value, state_schema[key], threshold=0.80)
            if best_match is None:
                raise ValueError(f"Invalid choice '{value}' for state variable '{key}'. Valid choices: {state_schema[key]}")
            if score < 1.0:
                logger.warning(f"Initial state: semantic matched '{value}' → '{best_match}' (score: {score:.3f}) for '{key}'")
            initial_state[key] = best_match

        # Ensure all schema keys are present
        for key in state_schema:
            if key not in initial_state:
                raise ValueError(f"Missing state variable: {key}")

        return initial_state

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing initial state: {e}\nRaw response: {response}")
        raise e


@on_exception(expo, Exception, max_tries=5)
def sample_state_updates(
    llm_config, start_date, user_profile, num_months, current_date, end_date,
    num_changes_per_period, max_changes_per_state,
    state_schema, latest_state, prior_updates, update_cnts,
    remaining_steps=10, total_steps=10, error_hist=(), lang="en"
):
    """
    Sample updates to the user's state variables for a period based on their current profile and the current state.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc
        start_date (str): The start date of the simulation in YYYY-MM-DD format.
        user_profile (str): The user's profile containing personal information at the start of the simulation.
        num_months (int): The number of months in the current period for which updates are to be made.
        current_date (datetime): The current date at the start of the period.
        end_date (datetime): The end date of the current period.
        num_changes_per_period (int): The expected number of state variables to change in this period.
        max_changes_per_state (int): The maximum number of times a single state variable can be changed across all periods.
        state_schema (dict): A schema defining the state variables and their possible values.
        latest_state (dict): The most recent state of the user's personal information variables.
        prior_updates (list): A list of prior updates made to the user's state variables.
        update_cnts (dict): A dictionary tracking how many times each state variable has been updated.
        remaining_steps (int): The remaining number of steps to simulate updates for.
        total_steps (int): The total number of steps in the simulation.
        error_hist (tuple): A tuple containing any previous error information for reflection.
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        dict: A dictionary containing updates and a summary of the corresponding period.
            - "updates": A dictionary where each key is an updated state variable and each value is the updated value.
            - "period_start": A string representing the start date of the period (YYYY-MM-DD).
            - "period_end": A string representing the end date of the period (YYYY-MM-DD).
            - "period_summary": A string summarizing the changes and context for the updates.
    """
    current_date_str = current_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    current_step = total_steps - remaining_steps + 1

    env_prompts = load_prompts("env", lang=lang, escape=False)
    prompt = env_prompts["sample_state_updates_prompt"].format(
        num_months=num_months,
        current_step=current_step,
        total_steps=total_steps,
        remaining_steps=remaining_steps,
        current_date_str=current_date_str,
        end_date_str=end_date_str,
        start_date=start_date,
        user_profile=user_profile,
        state_schema_json=json.dumps(state_schema, indent=2, ensure_ascii=False),
        latest_state_json=json.dumps(latest_state, indent=2, ensure_ascii=False),
        prior_updates_json=json.dumps(prior_updates, indent=2, ensure_ascii=False),
        max_changes_per_state=max_changes_per_state,
        update_cnts_json=json.dumps(update_cnts, indent=2, ensure_ascii=False),
        num_changes_per_period=num_changes_per_period,
    )

    messages = [{"role": "user", "content": prompt}]
    if len(error_hist) >= 4:
        raise ValueError("Too many errors encountered, stopping further updates.")
    if error_hist:
        error = error_hist[-1]
        messages.extend([
            {"role": "assistant", "content": error["response"]},
            {"role": "user", "content": f"Please try again to fix the error {error['info']} in your response."}
        ])
    response = call_llm(messages, llm_config, json=True)

    try:
        error_info = None
        update_info = json.loads(response)
        updates = update_info["updated"]
        # check number of changes
        if not (num_changes_per_period - 1 <= len(updates) <= num_changes_per_period + 1):
            error_info = {
                "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                "info": f"Number of changes {len(updates)} not in expected range [{num_changes_per_period - 1}, {num_changes_per_period + 1}]"
            }
            update_info = sample_state_updates(
                llm_config, start_date, user_profile, num_months, current_date, end_date,
                num_changes_per_period, max_changes_per_state,
                state_schema, latest_state, prior_updates, update_cnts,
                remaining_steps, total_steps, error_hist + (error_info,), lang
            )

        # Validate each update with semantic matching
        for state_var, new_value in updates.items():
            error_info = None

            if state_var not in state_schema:
                error_info = {
                    "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                    "info": f"Invalid state variable '{state_var}' in updates"
                }
            else:
                # Semantic matching
                best_match, score = find_best_semantic_match(new_value, state_schema[state_var], threshold=0.80)
                if best_match is None:
                    error_info = {
                        "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                        "info": f"Invalid value '{new_value}' for state variable '{state_var}'. Valid choices: {state_schema[state_var]}"
                    }
                else:
                    if score < 1.0:
                        logger.warning(f"Semantic matched '{new_value}' → '{best_match}' (score: {score:.3f}) for '{state_var}'")
                    updates[state_var] = best_match
                    if latest_state[state_var] == best_match:
                        error_info = {
                            "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                            "info": f"State variable '{state_var}' is not actually changing from '{best_match}'"
                        }

            if error_info:
                update_info = sample_state_updates(
                    llm_config, start_date, user_profile, num_months, current_date, end_date,
                    num_changes_per_period, max_changes_per_state,
                    state_schema, latest_state, prior_updates, update_cnts,
                    remaining_steps, total_steps, error_hist + (error_info,), lang
                )
                return update_info
        update_info["period_end"] = end_date_str
        update_info["period_start"] = current_date_str
        return update_info
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing state updates: {e}\nRaw response: {response}")
        raise e


@on_exception(expo, Exception, max_tries=5)
def elaborate_state_updates(
    llm_config, start_date, user_profile, current_state, updates, state_schema, lang="en"
):
    """Elaborate on the state updates by providing triggers for each change.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc.
        start_date (str): The date when the user's profile was created, in YYYY-MM-DD format.
        user_profile (str): The user's profile containing personal information at the start of the simulation.
        current_state (dict): The current state of the user's personal variables.
        updates (dict): A dictionary containing the latest state updates.
            - "period_start": The start date of the period for which updates are made (YYYY-MM-DD).
            - "period_end": The end date of the period for which updates are made (YYYY-MM-DD).
            - "period_summary": A string summarizing the changes and context for the updates.
            - "updated": The latest state updates.
            - "old": The previous state of the user's personal variables before the latest updates.
        state_schema (dict): A schema defining the state variables and their possible values.
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        list: A list of dictionaries representing the elaborated events and their relationships to state changes.
            - "event": A description of the event that serves as a trigger/implication for the state change.
            - "states": A list of state variables that are affected by this event.
    """
    states_not_updated = {k: v for k, v in current_state.items() if k not in updates["updated"]}

    # Extract state changes for context
    state_changes = []
    for state_var, new_value in updates["updated"].items():
        old_value = updates["old"][state_var]
        state_changes.append({
            "variable": state_var,
            "from": old_value,
            "to": new_value,
            "possible_values": state_schema[state_var]
        })

    env_prompts = load_prompts("env", lang=lang, escape=False)
    prompt = env_prompts["elaborate_state_updates_prompt"].format(
        start_date=start_date,
        user_profile=user_profile,
        period_start=updates["period_start"],
        period_end=updates["period_end"],
        period_summary=updates["period_summary"],
        state_changes_json=json.dumps(state_changes, indent=2, ensure_ascii=False),
        states_not_updated_json=json.dumps(states_not_updated, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)

    try:
        result = json.loads(response)
        events = result["events"]
        covered_states = set()

        # Validate that all mentioned states are in the updates
        for event_info in events:
            for state_var in event_info["states"]:
                if state_var not in updates["updated"]:
                    raise ValueError("Event mentions state variable that wasn't updated")
                covered_states.add(state_var)

        if len(covered_states) != len(state_changes):
            raise ValueError(
                f"Not all state changes are covered in the events. Covered: {covered_states}, Expected: {set(updates['updated'].keys())}"
            )

        return events

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing event elaboration: {e}\nRaw response: {response}")
        raise e
