import re


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _parse_answer(text: str) -> str | None:
    matches = re.findall(r"Answer\s*:\s*(.+)", text)
    if not matches:
        return None
    return matches[-1].strip()


def reward_fn_baseline(completions: list[list[dict]], **kwargs) -> list[float]:
    solutions = kwargs["solution"]
    rewards = []

    for completion, solution in zip(completions, solutions):
        text = completion[0]["content"] if completion else ""
        parsed = _parse_answer(text)

        if parsed is not None:
            correct = _normalize(parsed) == _normalize(solution)
        else:
            correct = _normalize(text.strip()) == _normalize(solution)

        rewards.append(1.0 if correct else 0.0)

    return rewards
