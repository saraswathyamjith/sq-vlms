SYSTEM_PROMPT = (
    "You are given an image and a question. Before answering, generate "
    "a series of sub-questions to help you analyze the image carefully. "
    "For each sub-question, look at the image and provide an answer. "
    "Then give your final answer.\n\n"
    "IMPORTANT: Your Final Answer must be a short phrase (1-5 words). "
    "Do not write a full sentence — just the answer itself.\n\n"
    "Format:\n"
    "Sub-question 1: [question]\n"
    "Answer 1: [answer]\n"
    "Sub-question 2: [question]\n"
    "Answer 2: [answer]\n"
    "...\n"
    "Final Answer: [short answer]"
)


def build_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
