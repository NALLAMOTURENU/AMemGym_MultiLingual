"""Centralized prompt loading utility with brace-escaping support.

Design notes
------------
* **Assistant prompts** (``assistants.json``) are stored in *raw* form — literal
  ``{}`` braces for JSON examples, and the valid runtime placeholders
  (``{current_memories}``, ``{conversation}``, ``{memory_types_section}``) as
  plain single-brace markers.  ``load_prompts`` applies ``escape_prompt`` so the
  returned strings are safe for Python's ``str.format()``.

* **The template** (``in_context_memory_update_prompt_template``) is stored in
  *Python-ready* form — ``{{current_memories}}`` (double-braced) so it survives a
  first ``.format(memory_types_section=…)`` call and remains usable for the
  subsequent ``.format(current_memories=…, conversation=…)`` call.
  The escaping pass is effectively a no-op for this key.

* **Env prompts** (``env.json``) are stored in *Python-ready* form with ``{{``
  for literal JSON braces and ``{variable}`` for format placeholders.  Call
  ``load_prompts('env', lang, escape=False)`` and use ``.format(**kwargs)``
  directly.

* Language fall-back: if a ``zh`` file is missing, the loader transparently
  falls back to ``en``.
"""

import json
import os
import re

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "prompts")

# Placeholders that must survive the brace-escaping pass for assistant prompts.
# These are the valid runtime format keys used by _update_memory and template
# assembly respectively.
_PROTECTED = ["current_memories", "conversation", "memory_types_section"]


def escape_prompt(prompt: str, protected: list[str] | None = None) -> str:
    """Escape single braces in *prompt*, preserving the listed *protected* placeholders.

    All ``{`` / ``}`` that are not already doubled (i.e. ``{{`` / ``}}``) are
    escaped to ``{{`` / ``}}``, making the string safe for ``str.format()``.
    The *protected* names are temporarily swapped out before escaping and
    restored afterwards so they survive as valid ``{name}`` placeholders.

    Parameters
    ----------
    prompt:
        The raw prompt string to escape.
    protected:
        Names of placeholders to preserve.  Defaults to
        ``["current_memories", "conversation", "memory_types_section"]``.

    Returns
    -------
    str
        The brace-escaped prompt, ready for ``str.format(current_memories=…)``.
    """
    if protected is None:
        protected = _PROTECTED

    tokens: dict[str, str] = {}
    for i, name in enumerate(protected):
        token = f"___PROMPT_PLACEHOLDER_{i}___"
        tokens[token] = name
        # Use regex so we only match genuinely single-braced occurrences: {name}
        # where { is NOT preceded by another { and } is NOT followed by another }.
        # This avoids matching {name} that is nested inside {{name}}.
        prompt = re.sub(
            r"(?<!\{)\{" + re.escape(name) + r"\}(?!\})",
            token,
            prompt,
        )

    prompt = re.sub(r"(?<!\{)\{(?!\{)", "{{", prompt)
    prompt = re.sub(r"(?<!\})\}(?!\})", "}}", prompt)

    for token, name in tokens.items():
        prompt = prompt.replace(token, "{" + name + "}")

    return prompt


def load_prompts(file_name: str, lang: str = "en", escape: bool = True) -> dict:
    """Load a prompt JSON file for the given language, with optional brace escaping.

    Parameters
    ----------
    file_name:
        Base name of the JSON file without extension, e.g. ``"assistants"`` or
        ``"env"``.
    lang:
        Language code, e.g. ``"en"`` or ``"zh"``.  Falls back to ``"en"`` if
        the requested language file does not exist.
    escape:
        When ``True`` (default), apply :func:`escape_prompt` to every string
        value in the loaded dict.  Pass ``False`` for env prompts that are
        already stored in Python-ready form (``{{`` for literal braces).

    Returns
    -------
    dict
        Mapping of prompt keys to prompt strings (optionally brace-escaped).
    """
    assets_dir = os.path.normpath(_ASSETS_DIR)
    lang_path = os.path.join(assets_dir, lang, f"{file_name}.json")
    fallback_path = os.path.join(assets_dir, "en", f"{file_name}.json")

    if os.path.exists(lang_path):
        path = lang_path
    elif os.path.exists(fallback_path):
        path = fallback_path
    else:
        raise FileNotFoundError(
            f"Prompt file '{file_name}.json' not found for lang='{lang}' "
            f"and no English fallback exists at '{fallback_path}'."
        )

    with open(path, encoding="utf-8") as fh:
        prompts: dict = json.load(fh)

    if escape:
        return {
            k: escape_prompt(v) if isinstance(v, str) else v
            for k, v in prompts.items()
        }
    return prompts
