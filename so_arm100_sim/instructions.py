"""Language instructions for the four tasks, with fixed train / held-out splits.

Shared by the imitation-learning and language-conditioned projects so that
"unseen paraphrase" means the same sentences in both. Every template has a
`{c}` slot for the cube colour. The held-out templates are never used to
generate training data; the held-out (task, colour) combinations are the
compositional split -- every task and every colour appears in training,
just not together.

The splits are lists, not sets, and in a fixed order: iteration order is part
of the contract because dataset generation and evaluation index into them.
"""
from . import scene

TASKS = ("reach", "push", "lift", "pick_place")

TRAIN_TEMPLATES = {
    "reach": [
        "move the gripper above the {c} cube",
        "hover over the {c} cube",
        "go to the {c} cube",
        "bring the gripper to the {c} block",
        "position the arm over the {c} cube",
    ],
    "push": [
        "push the {c} cube to the target",
        "slide the {c} cube onto the yellow marker",
        "move the {c} block to the target by pushing it",
        "push the {c} block onto the marker",
        "shove the {c} cube to the yellow spot",
    ],
    "lift": [
        "pick up the {c} cube",
        "lift the {c} cube",
        "grab the {c} block and raise it",
        "pick the {c} block up off the table",
        "raise the {c} cube into the air",
    ],
    "pick_place": [
        "put the {c} cube on the target",
        "place the {c} block on the yellow marker",
        "pick up the {c} cube and put it on the target",
        "move the {c} cube onto the marker",
        "carry the {c} block to the yellow spot",
    ],
}

HELDOUT_TEMPLATES = {
    "reach": [
        "reach toward the {c} cube",
        "get the gripper over the {c} one",
        "float the hand above the {c} block",
    ],
    "push": [
        "nudge the {c} cube over to the marker",
        "get the {c} block onto the yellow target by sliding it",
        "push the {c} one to the yellow circle",
    ],
    "lift": [
        "hoist the {c} block",
        "take the {c} cube and hold it up",
        "get the {c} one off the table",
    ],
    "pick_place": [
        "set the {c} cube down on the marker",
        "relocate the {c} block to the yellow target",
        "drop the {c} cube onto the yellow spot",
    ],
}

# Compositional split. Every task and every colour appears on the train side.
HELDOUT_COMBOS = [("lift", "blue"), ("push", "red"), ("pick_place", "green")]
TRAIN_COMBOS = [(t, c) for t in TASKS for c in scene.CUBE_COLORS
                if (t, c) not in HELDOUT_COMBOS]


def instructions(task, color, split="train"):
    """All sentences for one (task, colour) in one split."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}")
    if color not in scene.CUBE_COLORS:
        raise KeyError(f"unknown colour {color!r}")
    table = {"train": TRAIN_TEMPLATES, "heldout": HELDOUT_TEMPLATES}[split]
    return [t.format(c=color) for t in table[task]]


def all_instructions(split="train", combos=None):
    """[(task, colour, sentence), ...] in fixed order."""
    combos = TRAIN_COMBOS if combos is None else combos
    out = []
    for task, color in combos:
        for s in instructions(task, color, split):
            out.append((task, color, s))
    return out


def sample_instruction(task, color, rng, split="train"):
    opts = instructions(task, color, split)
    return opts[int(rng.integers(len(opts)))]
