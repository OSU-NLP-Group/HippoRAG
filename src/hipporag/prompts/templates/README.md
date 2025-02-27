# Information for templates folder:

- Treated as the module under the hipporag project.

- Each *.py (excluding `__init__.py`) corresponds to a prompt template and the corresponding filename (without file extension) will be used as the prompt template name (key to access the prompt template).

- Each *.py should define a variable `prompt_template` to store prompt template.

- A prompt template can be:
    - A str with or without ${}-like placeholders for filling values (will be converted with Template(...)) OR
    - A Template instance OR
    - A chat history (List[dict[str, Any]]) with each dict is like {"role": "system"/"user"/"assistant", "content": the above two option items}.


