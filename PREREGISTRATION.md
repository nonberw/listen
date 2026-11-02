# Pre-registration: When do robot policies learn to listen?

*Written and committed before any experiment was run. Later changes go in a dated
"Deviations" section at the bottom; nothing above it gets edited after the pilot.*

## Background

Several recent studies show that vision-language-action (VLA) policies often ignore their
instruction and execute the task they saw most in a given scene (LIBERO-Plus, arXiv
2510.13626; LIBERO-PRO, arXiv 2510.03827; LIBERO-CF, arXiv 2602.17659; "Encoded but Not in
Control", arXiv 2610.06235). The usual explanation is the structure of robot datasets: a scene
appears with only one or a few tasks, so the scene alone predicts the action, and language
carries no extra information. As far as we could find, this explanation has not been tested
causally. The existing work evaluates or probes pretrained models whose training data cannot be
changed. "Encoded but Not in Control" explicitly lists "training that explicitly require[s]
instructions to select among competing valid behaviors" as future work.

## Research question

Holding the total amount of data fixed, how does the share of **contrastive scenes**
(scenes demonstrated with several different instructions) control whether an imitation-learned,
language-conditioned policy follows its instruction?

## Setup

- **Robot and task.** The ARMageddon 6-DoF arm with a two-finger gripper in MuJoCo. Three
  colored blocks on a table (colors drawn from a pool of 4). Instruction: "pick up the
  `<color>` block", in one of several paraphrase templates. The success criterion is that the
  named block gets lifted.
- **Scenes.** 40 scene templates, each a fixed set of 3 colors at fixed nominal positions.
  Every episode adds a small jitter (+-2 cm, +-15 deg).
- **Trained targets.** A non-contrastive scene is always demonstrated with the same single
  target (k = 1). A contrastive scene is demonstrated with all 3 targets (k = 3), in equal
  shares. Across the non-contrastive scenes, every color is the target equally often, so that a
  global "always pick red" prior is useless. The only shortcuts left are memorizing
  scene -> target, or reading the instruction.
- **Independent variable.** The contrastive fraction f, from {0, 0.05, 0.1, 0.25, 0.5, 1.0}
  of the 40 scenes. The total number of demonstrations N is identical in every condition.
- **Expert.** Scripted, with privileged object poses, using the same IK and motion stack as
  YOINK, plus DART-style noise injection.
- **Policy.** A CNN on a 64x64 RGB image plus proprioception, and a frozen pretrained sentence
  encoder for the instruction. Language enters via FiLM. Action chunks are end-effector deltas
  plus the gripper command. The training recipe is identical across conditions.
- **Seeds.** 3 training seeds per condition (more if compute allows, added equally to all
  conditions).

## Metrics

- **Primary: counterfactual grounding rate (CGR).** In a *non-contrastive training scene*, ask
  for one of the two blocks that this scene was never demonstrated with. CGR is the fraction of
  episodes in which the first block the gripper commits to (the gripper closes within 3 cm of it,
  horizontally) is the instructed one. We also log the **default rate**: the share of episodes
  that commit to the scene's trained target instead.
- **Secondary.**
  1. Nominal success on trained instructions, to check that grounding doesn't come at the cost
     of skill.
  2. CGR in contrastive scenes, on held-out scene layouts, and on held-out paraphrases.
  3. Lift success for counterfactual instructions.

Each condition x seed is evaluated on a fixed set of 300 counterfactual episodes and
150 nominal episodes, with the same seeds for all conditions.

## Hypotheses

- **H1.** At f = 0, CGR in non-contrastive scenes stays low (below 30%), and the default rate is
  high.
- **H2.** CGR increases monotonically with f.
- **H3 (transfer, main question).** Contrastive data in *some* scenes raises grounding in the
  *other*, non-contrastive scenes. That is, at f in {0.1, 0.25}, CGR in non-contrastive scenes
  is significantly above f = 0.
- **H4.** Nominal success does not decrease with f by more than 10 percentage points.

## Analysis plan

- Per condition: the mean CGR across seeds, with 95% CIs from a bootstrap over seeds and
  episodes. Wilson intervals for per-seed rates.
- H2 and H3: a logistic regression of grounding on f (a categorical f effect, with a random
  intercept per seed if the fit allows; otherwise a fixed seed effect), plus a one-sided
  comparison of each f > 0 against f = 0 with Holm correction.
- We report f80, the smallest tested f where CGR >= 80%, if there is one.
- Every condition and every metric gets reported, including those that don't support the
  hypotheses.

## Pilot

Before the main run, a pilot at f = 0 and f = 1 checks that the policy learns the task at all.
The pilot may change the image size, training steps, network size and N, but not the
hypotheses, metrics or conditions. Pilot results are reported separately and are not used for
hypothesis tests.

## Deviations

*(none yet)*
