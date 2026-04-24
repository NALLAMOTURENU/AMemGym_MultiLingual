import json
from amemgym.utils import call_llm
from amemgym.utils.prompt_loader import load_prompts


def _generate_user_followup(llm_config, query, user_profile, state_schema, start_date, current_date, conversation_history, lang="en"):
    """
    Generate a follow-up response from the simulated user based on the agent's response.

    Args:
        llm_config (dict): Configuration for the LLM
        query (str): The original query that started the conversation
        user_profile (dict): User profile information
        state_schema (dict): Schema for state variables
        start_date (str): Start date of the simulation in YYYY-MM format
        current_date (str): Current date in YYYY-MM format
        conversation_history (list): Previous messages in the conversation
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        str: User's follow-up message
    """
    context = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history[-2:]])

    env_prompts = load_prompts("env", lang=lang, escape=False)
    followup_prompt = env_prompts["generate_user_followup_prompt"].format(
        start_date=start_date,
        user_profile_str=user_profile['formatted_str'],
        current_date=current_date,
        query=query,
        context=context,
        state_schema_json=json.dumps(state_schema, indent=2, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": followup_prompt}]
    response = call_llm(messages, llm_config, json=False).strip()

    return response


def sample_session_given_query(
    llm_config, query, agent, start_date, user_profile, current_date, state_schema, hist=None, max_rounds=10, lang="en"
):
    """
    Sample a session in a specific period between a simulated user and an AI agent, given a query.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc.
        query (str): The query to be answered by the agent.
        agent (Agent): The AI agent that will interact with the user.
        start_date (str): The start date of the simulation.
        user_profile (dict): The user profile containing background information.
        current_date (str): The current date in YYYY-MM format.
        state_schema (dict): Schema defining the structure of all state variables and possible values.
        hist (list, optional): Optional history of previous messages in the session.
        max_rounds (int): Maximum number of rounds in the session.
        lang (str): Language code for prompt selection (default: "en")

    Returns:
        list: A list of messages representing the session.
    """
    if hist is None:
        session_messages = []
        current_user_input = query
        init_num_rounds = 0
    else:
        session_messages = hist.copy()
        init_num_rounds = len(session_messages) // 2
        if init_num_rounds >= max_rounds:
            return session_messages
        current_user_input = _generate_user_followup(
            llm_config, query, user_profile, state_schema,
            start_date, current_date, session_messages, lang
        )

    for num_rounds in range(init_num_rounds, max_rounds):
        session_messages.append({"role": "user", "content": current_user_input})

        agent_response = agent.act(current_user_input)
        session_messages.append({"role": "assistant", "content": agent_response})

        if num_rounds < max_rounds - 1:
            user_followup = _generate_user_followup(
                llm_config, query, user_profile, state_schema,
                start_date, current_date, session_messages, lang
            )
            current_user_input = user_followup

    return session_messages
