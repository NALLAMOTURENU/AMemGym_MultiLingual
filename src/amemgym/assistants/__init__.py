from .awi import InContextMemAgent
from .native import NaiveAgent
from .mem0 import Mem0Agent
from .evolvable import EvolvableInContextAgent, EvolvableMem0Agent
import os

from dotenv import load_dotenv
load_dotenv()


def create_agent(agent_config, output_dir, item=None):
    # set llm inference parameters
    agent_config["llm_config"] |= {
        "base_url": agent_config["llm_config"].get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": agent_config["llm_config"].get("api_key") or os.environ.get("OPENAI_API_KEY")
    }  # fill in from environment variables if not provided

    # create agent based on type
    agent_type = agent_config["type"]
    if agent_type == "native":
        return NaiveAgent(agent_config["llm_config"])
    elif agent_type == "awi":
        return InContextMemAgent(agent_config)
    elif agent_type == "awi-hack":
        # with known info types (only for ablation study)
        assert item is not None, "the specific item is required for the hack setting"
        agent_config["info_types"] = list(item["state_schema"].keys())
        return InContextMemAgent(agent_config)
    elif agent_type == "awi-evolve":
        return EvolvableInContextAgent(agent_config)
    elif agent_type in ("rag", "awe"):
        local_mem_dir = os.path.join(output_dir, "latest_memories")
        return Mem0Agent(agent_config | {"local_mem_dir": local_mem_dir})
    elif agent_type in ("rag-evolve", "mem0-evolution"):
        local_mem_dir = os.path.join(output_dir, "latest_memories")
        return EvolvableMem0Agent(agent_config | {"local_mem_dir": local_mem_dir})
    # elif agent_type == "a-mem":
    #     local_mem_dir = os.path.join(output_dir, "latest_memories")
    #     return AMemAgent(agent_config | {"local_mem_dir": local_mem_dir})
    # elif agent_type == "nemori":
    #     local_mem_dir = os.path.join(output_dir, "latest_memories")
    #     return NemoriAgent(agent_config | {"local_mem_dir": local_mem_dir})
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
