SELF_GEN_SYSTEM_MSG = "You are an honest and helpful assistant."

SELF_QA_INTX = (
    "# System Instruction\n"
    "- The information provided is up-to-date information and/or the user instruction.\n"
    "- When the provided information is not relevant to the question, ***ignore*** it and answer the question based on your knowledge.\n"
    "- If the provided information is related to the question, incorporate it in your response.\n"
    "- If the provided information is an instruction, follow the instruction carefully.\n"
    "\n---\n\n"
    "# User Input\n"
)

PRE_CTX = "# Provided Information\n"

QA_PROMPT_TEMPLATE = PRE_CTX + "{context}\n\n---\n\n" + SELF_QA_INTX + "{question}"
PROMPT_TEMPLATE = "{context}\n\n{question}"
