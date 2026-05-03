import re


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _check_correct(prediction: str, solution: str) -> bool:
    pred_norm = _normalize(prediction)
    sol_norm = _normalize(solution)
    if not sol_norm:
        return False
    if pred_norm == sol_norm:
        return True
    return sol_norm in pred_norm


def _has_sub_questions(text: str) -> bool:
    return bool(
        re.search(r"Sub-question\s+\d+\s*:", text)
        and re.search(r"Answer\s+\d+\s*:", text)
    )


def _parse_final_answer(text: str) -> str | None:
    matches = re.findall(r"Final Answer\s*:\s*(.+)", text)
    if not matches:
        return None
    return matches[-1].strip()


def reward_fn_direct(completions: list[list[dict]], **kwargs) -> list[float]:
    solutions = kwargs["solution"]
    rewards = []

    for completion, solution in zip(completions, solutions):
        text = completion[0]["content"] if completion else ""

        matches = re.findall(r"Answer\s*:\s*(.+)", text)
        if matches:
            parsed = matches[-1].strip()
        else:
            parsed = text.strip().split("\n")[0].strip()

        if parsed and _check_correct(parsed, solution):
            word_count = len(parsed.split())
            rewards.append(1.0 if word_count <= 5 else 0.5)
        else:
            rewards.append(0.0)

    return rewards


def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
    solutions = kwargs["solution"]
    rewards = []

    for completion, solution in zip(completions, solutions):
        text = completion[0]["content"] if completion else ""

        has_format = _has_sub_questions(text)
        parsed_answer = _parse_final_answer(text)

        if has_format and parsed_answer is not None:
            correct = _check_correct(parsed_answer, solution)
            if correct:
                word_count = len(parsed_answer.split())
                if word_count <= 5:
                    reward = 1.0
                else:
                    reward = 0.5
            else:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards
