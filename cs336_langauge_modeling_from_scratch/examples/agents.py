"""Agent trajectories as training data: segments, loss masks and verifiable rewards.

An agent episode alternates model-generated text (reasoning, tool calls, final
answer) with environment text (tool results). Only model-generated tokens are
actions, so only they enter the policy-gradient or SFT loss. This module builds
that mask for a toy calculator environment, checks it against a byte-level
tokenization, and scores episodes with an outcome reward plus a format check.
"""
import ast
from fractions import Fraction
import json
import operator
import re

TOOL_CALL = re.compile(r"<tool>(.*?)</tool>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression, max_digits=30, max_length=200):
    """The environment: + - * / on integers via an AST walk (no eval, no powers), result or error string.

    Model output is untrusted: overly long input is rejected before parsing (deep nesting would
    otherwise exhaust the recursion limit), and every constant and result is bounded in size.
    Arithmetic is exact (Fraction), so large integer quotients are not rounded through float.
    """
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) >= 10 ** max_digits:
                raise OverflowError
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            result = OPERATORS[type(node.op)](value(node.left), value(node.right))
            if abs(result) >= 10 ** max_digits:
                raise OverflowError
            return result
        raise ValueError
    if len(expression) > max_length:
        return "error: expression too long"
    try:
        result = value(ast.parse(expression.strip(), mode="eval").body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "error: unsupported expression"
    except (ZeroDivisionError, OverflowError) as exc:
        return f"error: {type(exc).__name__}"
    return str(result.numerator) if result.denominator == 1 else f"{float(result):.6g}"


def run_episode(policy_turns, max_calls=4):
    """policy_turns: the model's successive messages. Returns segments [(role, text)]."""
    segments = []
    for turn in policy_turns:
        segments.append(("model", turn))
        call = TOOL_CALL.search(turn)
        if ANSWER.search(turn) or call is None:
            break
        if sum(role == "tool" for role, _ in segments) >= max_calls:
            break
        segments.append(("tool", f"<result>{calculator(call.group(1))}</result>"))
    return segments


def byte_tokens_and_mask(prompt, segments):
    """Byte-level token IDs for the whole episode and a 0/1 mask that is 1 only on model bytes."""
    ids, mask = list(prompt.encode()), [0] * len(prompt.encode())
    for role, text in segments:
        data = text.encode()
        ids.extend(data)
        mask.extend([1 if role == "model" else 0] * len(data))
    return ids, mask


def reward(segments, target):
    """Outcome reward 1 for a correct final answer; format reward checks the answer tag exists."""
    final = segments[-1][1] if segments and segments[-1][0] == "model" else ""
    match = ANSWER.search(final)
    formatted = match is not None
    correct = formatted and match.group(1).strip() == str(target)
    calls = sum(role == "tool" for role, _ in segments)
    errors = sum(role == "tool" and "error" in text for role, text in segments)
    return dict(reward=float(correct), format=float(formatted), tool_calls=calls, tool_errors=errors)


def demo():
    prompt = "Q: What is (17 * 23) + 389? Use the calculator.\n"
    good = ["<tool>17 * 23</tool>", "391 so far. <tool>391 + 389</tool>", "<answer>780</answer>"]
    sloppy = ["<tool>17 x 23</tool>", "<answer>781</answer>"]
    results = {}
    for name, turns in [("good", good), ("sloppy", sloppy)]:
        segments = run_episode(turns)
        ids, mask = byte_tokens_and_mask(prompt, segments)
        results[name] = dict(segments=segments, tokens=len(ids), loss_tokens=sum(mask),
                             tool_tokens=sum(len(t.encode()) for r, t in segments if r == "tool"), **reward(segments, 780))
    return results


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
