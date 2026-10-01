# Fixed-budget allocation replay

Frozen before running `analyze.py`.

- Pool: the 282 `A_current` candidates from the frozen H2 proof.
- Budget: the production repeat budget of 2,244 candidate-observations; the
  one-pass screen is common to every arm.
- Evaluation: four-fold cross-validation over the sixteen complete scorer
  draws. Each fold trains/prunes on twelve draws and evaluates on the untouched
  four draws against the eight reserved public competitor fields.
- Fleet: ten pairwise-distinct sequences at the live 0.94 identity cap.
- Primary outcome: individual wins per entry, because every ticket that beats
  the external field is valuable. Secondary outcomes are top-10 per entry and
  probability that at least one fleet member beats a field.
- Allocation arms: the live 4/4/3 successive-elimination shape; delayed-prune
  5/6, 6/4, and 7/2 shapes at the same observation budget; and a near-budget
  uniform-eight arm (twelve extra observations, but three fewer model loads).
- Ranking arms: the live internal-tournament probability and the corrected
  empirical probability of an individual entry beating the external field.
- No production deployment is authorized by this replay alone. A change needs
  a consistent held-out advantage, no loss of publishability, and a wall-clock
  fit in the existing epoch/GPU budget.
