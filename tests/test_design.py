"""The design is the experiment: these tests pin down that conditions differ only where intended."""

import numpy as np
import pytest
import torch

from listen.design import EVAL_SCENES, FRACTIONS, Design, HELDOUT_TEMPLATES, TRAIN_TEMPLATES
from listen.policy import CHUNK, Policy, random_shift

D = Design()


def test_design_is_deterministic():
    other = Design()
    assert other.default_target == D.default_target
    assert other.contrastive_order == D.contrastive_order
    for a, b in zip(other.train_scenes, D.train_scenes):
        assert a.colors == b.colors and np.array_equal(a.spots, b.spots)


@pytest.mark.parametrize("f", FRACTIONS)
def test_every_condition_has_the_same_amount_of_data(f):
    assert sum(c for *_, c in D.demo_plan(f)) == 1200


def test_contrastive_sets_are_nested():
    sets = [D.contrastive(f) for f in FRACTIONS]
    for small, big in zip(sets, sets[1:]):
        assert small <= big


def test_eval_scenes_stay_non_contrastive_below_one():
    for f in FRACTIONS[:-1]:
        assert not set(EVAL_SCENES) & D.contrastive(f)


def test_no_global_color_prior():
    counts = list(D.default_color_counts.values())
    assert max(counts) - min(counts) <= 2


def test_non_contrastive_scene_uses_only_its_default():
    for s, t, _ in D.demo_plan(0.0):
        assert t == D.default_target[s]


def test_heldout_paraphrases_never_used_in_training():
    rng = np.random.default_rng(0)
    seen = {D.instruction("red", rng) for _ in range(500)}
    assert not seen & {t.format(c="red") for t in HELDOUT_TEMPLATES}
    assert seen == {t.format(c="red") for t in TRAIN_TEMPLATES}


def test_scene_spots_are_separated():
    for s in D.train_scenes + D.heldout_scenes:
        d = [np.linalg.norm(s.spots[i] - s.spots[j]) for i in range(3) for j in range(i)]
        assert min(d) > 0.09 and len(set(s.colors)) == 3


def test_policy_shapes_and_language_matters_at_init():
    torch.manual_seed(0)
    p = Policy()
    img = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    prop = torch.zeros(2, 4)
    l1, l2 = torch.randn(2, 384), torch.randn(2, 384)
    out1, out2 = p(img, prop, l1), p(img, prop, l2)
    assert out1.shape == (2, CHUNK, 4)
    assert not torch.allclose(out1, out2)  # the language pathway is wired in


def test_random_shift_keeps_shape_and_content_range():
    img = torch.randint(0, 255, (4, 64, 64, 3), dtype=torch.uint8)
    out = random_shift(img)
    assert out.shape == img.shape and out.dtype == torch.uint8
    zero = random_shift(img, pad=0)
    assert torch.equal(zero, img)


def test_temporal_ensemble_votes_across_chunks():
    from listen.policy import TemporalEnsemble

    ens = TemporalEnsemble(decay=0.0)
    first = np.zeros((CHUNK, 4))
    first[1, 3] = 1.0                      # chunk 1 predicts "close" for the next step
    np.testing.assert_allclose(ens(first), first[0])
    second = np.full((CHUNK, 4), -1.0)     # chunk 2 says "open" for its first step
    out = ens(second)
    np.testing.assert_allclose(out[3], 0.0)  # equal weights: (+1 + -1) / 2
