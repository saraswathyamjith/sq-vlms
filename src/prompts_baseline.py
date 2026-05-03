SYSTEM_PROMPT_BASELINE = (
    "You are given an image and a question. "
    "Answer the question about the image.\n\n"
    "Format:\n"
    "Answer: [answer]"
)


def build_prompt_baseline(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
        {"role": "user", "content": question},
    ]
