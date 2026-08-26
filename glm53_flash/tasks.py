"""Deterministic synthetic coding corpus and frozen executable evaluations."""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Family:
    name: str
    arguments: str
    descriptions: tuple[str, ...]
    body: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    family: str
    entry_point: str
    prompt: str
    reference_completion: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]

    @property
    def reference_source(self) -> str:
        return self.prompt + self.reference_completion


FAMILIES = (
    Family("increment", "x", ("Return x plus one.", "Increase x by one."), "\n    return x + 1\n", (((-3,), -2), ((0,), 1), ((7,), 8))),
    Family("double", "x", ("Return two times x.", "Double the input."), "\n    return x * 2\n", (((-4,), -8), ((0,), 0), ((6,), 12))),
    Family("square", "x", ("Return x squared.", "Multiply x by itself."), "\n    return x * x\n", (((-4,), 16), ((0,), 0), ((5,), 25))),
    Family("absolute", "x", ("Return the absolute value of x.", "Make a negative x positive."), "\n    return -x if x < 0 else x\n", (((-7,), 7), ((0,), 0), ((9,), 9))),
    Family("nonnegative", "x", ("Clamp x to at least zero.", "Return zero when x is negative."), "\n    return x if x > 0 else 0\n", (((-5,), 0), ((0,), 0), ((8,), 8))),
    Family("even", "x", ("Return whether x is even.", "Check divisibility by two."), "\n    return x % 2 == 0\n", (((-3,), False), ((0,), True), ((8,), True))),
    Family("reverse", "text", ("Return text in reverse order.", "Reverse the string."), "\n    return text[::-1]\n", ((('',), ''), (('abc',), 'cba'), (('level',), 'level'))),
    Family("list_sum", "values", ("Return the sum of values.", "Add every number in the list."), "\n    return sum(values)\n", ((([],), 0), (([1, 2, 3],), 6), (([-2, 5],), 3))),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}


def _name(rng: random.Random, prefix: str) -> str:
    suffix = "".join(rng.choice(string.ascii_lowercase) for _ in range(7))
    return f"{prefix}_{suffix}"


def make_task(family: Family, *, split: str, index: int, seed: int) -> CodingTask:
    rng = random.Random((seed + 1) * 1_000_003 + index * 9176 + sum(map(ord, family.name)))
    entry = _name(rng, family.name)
    description = family.descriptions[rng.randrange(len(family.descriptions))]
    prompt = f"# Complete this Python function.\n# {description}\ndef {entry}({family.arguments}):"
    return CodingTask(
        task_id=f"{split}-{family.name}-{index:03d}",
        family=family.name,
        entry_point=entry,
        prompt=prompt,
        reference_completion=family.body,
        cases=family.cases,
    )


def frozen_tasks(split: str, *, per_family: int = 4) -> list[CodingTask]:
    seeds = {"dev": 1701, "final": 2909, "rl": 3907, "confirm": 8123}
    if split not in seeds:
        raise ValueError(f"unknown split: {split}")
    tasks = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(per_family):
            tasks.append(make_task(family, split=split, index=index, seed=seeds[split] + family_index * 53))
    return tasks


def pretraining_text(index: int, *, seed: int = 42) -> str:
    rng = random.Random(seed * 10_000_019 + index)
    family = FAMILIES[rng.randrange(len(FAMILIES))]
    task = make_task(family, split="pretrain", index=index, seed=seed)
    return task.reference_source
