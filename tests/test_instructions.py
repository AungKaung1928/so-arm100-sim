from so_arm100_sim import instructions as I
from so_arm100_sim import scene


def test_every_task_and_colour_is_covered_in_training():
    tasks = {t for t, _ in I.TRAIN_COMBOS}
    colours = {c for _, c in I.TRAIN_COMBOS}
    assert tasks == set(I.TASKS) and colours == set(scene.CUBE_COLORS)
    assert not set(I.TRAIN_COMBOS) & set(I.HELDOUT_COMBOS)
    assert len(I.TRAIN_COMBOS) + len(I.HELDOUT_COMBOS) == len(I.TASKS) * len(scene.CUBE_COLORS)


def test_templates_have_colour_slot_and_are_disjoint():
    for task in I.TASKS:
        tr, ho = I.TRAIN_TEMPLATES[task], I.HELDOUT_TEMPLATES[task]
        assert all("{c}" in t for t in tr + ho)
        assert not set(tr) & set(ho)
        assert len(tr) >= 5 and len(ho) >= 3


def test_instruction_lists_are_deterministic_and_ordered():
    a = I.all_instructions("train")
    b = I.all_instructions("train")
    assert a == b
    assert len(a) == len(I.TRAIN_COMBOS) * 5
    assert I.instructions("lift", "red")[0] == "pick up the red cube"


def test_heldout_sentences_never_appear_in_train():
    train = {s for _, _, s in I.all_instructions("train", I.TRAIN_COMBOS + I.HELDOUT_COMBOS)}
    held = {s for _, _, s in I.all_instructions("heldout", I.TRAIN_COMBOS + I.HELDOUT_COMBOS)}
    assert not train & held
