"""Error type aware tier allocation.

Why this exists
---------------
The original rule (build_tiers_from_ranking in application_aware_qec.py)
sorts logical qubits by combined_mse = (z_mse + x_mse) / 2 and gives the
top qubit the Shor code, the next two the repetition code, and the last
one nothing. That ignores WHAT KIND of error each qubit suffers.

The repetition code used here is the three qubit bit flip code. Reading
_repetition_encode and _repetition_decode in application_aware_qec.py:
  * it corrects X (bit flip) errors, which show up in the <Z> output,
    that is z_mse
  * it does NOT correct Z (phase flip) errors, which show up in the <X>
    output, that is x_mse, and because a Z error on any one of the three
    wires becomes a logical phase error, the exposure is roughly three
    times larger than for a bare qubit
So putting the repetition code on a qubit whose error is mostly in <X>
can make that qubit worse than leaving it bare, and leaving a qubit whose
error is mostly in <Z> bare throws away the one thing the code is good at.
The Shor code corrects both kinds to first order.

Scoring used here (first order, in the same MSE units as the ranking):
  bare        benefit 0
  repetition  benefit z_mse - rep_phase_penalty * x_mse
  shor        benefit z_mse + x_mse
The tier assignment with the highest total benefit, under the fixed
budget of n_shor + n_repetition + n_bare qubits, is returned. The
penalty default of 2.0 comes from the three wire argument above (phase
exposure goes from 1x to about 3x, so net extra 2x). It is not tuned on
any results.

IMPORTANT for the paper: this rule was written AFTER seeing that the
simple ranking rule failed on several subjects. Report it as a revised
rule, report the simple rule as the baseline, and evaluate the revised
rule on subjects that were not used to design it.
"""

from itertools import permutations


def build_tiers_error_aware(ranking_df, n_shor=1, n_repetition=2, n_bare=1, rep_phase_penalty=2.0):
    """ranking_df needs the columns logical_qubit, z_mse, x_mse, as saved by
    run_sensitivity_ranking.py. Returns {logical_qubit: tier_name}.
    """
    total = n_shor + n_repetition + n_bare
    if total != len(ranking_df):
        raise ValueError(
            f"tier counts must sum to the number of logical qubits, "
            f"got {n_shor}+{n_repetition}+{n_bare}={total}, expected {len(ranking_df)}"
        )

    qubits = sorted(int(q) for q in ranking_df["logical_qubit"])
    z = {int(r["logical_qubit"]): float(r["z_mse"]) for _, r in ranking_df.iterrows()}
    x = {int(r["logical_qubit"]): float(r["x_mse"]) for _, r in ranking_df.iterrows()}

    def benefit(q, tier):
        if tier == "shor":
            return z[q] + x[q]
        if tier == "repetition":
            return z[q] - rep_phase_penalty * x[q]
        return 0.0

    labels = ["shor"] * n_shor + ["repetition"] * n_repetition + ["bare"] * n_bare
    best_score, best_perm = None, None
    for perm in sorted(set(permutations(labels))):
        score = sum(benefit(q, t) for q, t in zip(qubits, perm))
        if best_score is None or score > best_score:
            best_score, best_perm = score, perm

    return {q: t for q, t in zip(qubits, best_perm)}