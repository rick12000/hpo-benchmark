# ShaRP Rank QoI: Shapley Values, Efficiency, and the Baseline Question

**Relevant code:** `hpobench/learning_to_rank/explainability.py` → `compute_shap_values`  
**Library:** `xai-sharp` (ShaRP)  
**Paper:** Pliatsika et al., *ShaRP: Explaining Rankings and Preferences with Shapley Values*, PVLDB 18(11): 4131–4143, 2025. [doi:10.14778/3749646.3749682](https://doi.org/10.14778/3749646.3749682)

---

## 1. Background: Why Standard SHAP Is Insufficient for Ranking

Standard SHAP (Lundberg & Lee, 2017) measures how much a feature shifts a model's **output score**. In a ranking task the relevant outcome is not the score but the **rank position** — an item's ordinal place among its competitors. These two quantities are related but not equivalent: a feature can increase a score without changing the rank (because all other scores shift similarly), and a small score change can cause a large rank jump near a boundary.

The ShaRP paper motivates this with a concrete example (Fig. 1 in the paper): two very different scoring functions `f = 0.4·gpa + 0.4·sat + 0.2·essay` and `g = 1.0·essay` produce almost identical rankings of 8 college applicants. From a score-SHAP perspective `essay` has low importance (weight 0.2 in `f`); from a rank perspective it is the dominant feature because it is the only discriminator between top-4 and bottom-4 candidates.

---

## 2. Shapley Values: The Core Idea

Given `d` features (players) and a scalar-valued payoff function `v`, the **Shapley value** of feature `j` is the weighted average of its marginal contributions across all possible coalitions `S ⊆ {1,...,d} \ {j}`:

```
φ_j = Σ_{S ⊆ {1,...,d}\{j}}  [|S|! (d-|S|-1)! / d!]  ·  [v(S ∪ {j}) − v(S)]
```

The weight `|S|! (d-|S|-1)! / d!` equals `1 / [C(d-1, |S|) · d]` and ensures every ordering of features is treated equally — feature `j`'s contribution is measured in every position it could occupy in a random arrival order, then averaged.

**Key axioms** that Shapley values satisfy:
- **Efficiency**: `Σ_j φ_j = v(all features) − v(no features)`
- **Symmetry**: two features with identical contributions get equal values
- **Dummy**: a feature that never changes the payoff gets value 0
- **Additivity**: values from two independent games add up

---

## 3. The Rank QoI

ShaRP replaces the score payoff with a **rank-specific Quantity of Interest (QoI)**. For an item `v` in dataset `D`, the rank QoI for coalition `S` is the expected rank of `v` when the features in `S` are randomised:

```
QoI(S) = E_{u ~ D\{v}} [ rank of v^{A\S} u^S in D' ]
```

where `v^{A\S} u^S` is a composite item that keeps `v`'s values for features outside `S` and takes `u`'s values for features in `S`. `D' = D \ {v} ∪ {composite}` is the dataset with `v` replaced by the composite.

Intuitively: "if we only knew the features in `S` for item `v`, and drew the remaining features randomly from the population, what rank would `v` get on average?"

The **iota function** (the marginal contribution of feature `j` given coalition `S`) is then:

```
ι(j, S) = −1 · [QoI(S ∪ {j}) − QoI(S)]
```

The `−1` multiplier accounts for the fact that **lower rank is better**: adding feature `j`'s true value to the coalition should lower the expected rank number, so without the sign flip the contribution would be negative for helpful features. With it, a **positive Shapley value means the feature improves the item's rank** (moves it closer to rank 1).

---

## 4. Computing a Marginal Contribution: Worked Example

Consider 4 tuners competing on a benchmark. The model assigns scores via a simplified rule `score = 2·A + B`:

| Tuner | A | B | score | rank |
|-------|---|---|-------|------|
| T1    | 3 | 1 |   7   |  1   |
| T2    | 2 | 3 |   7   |  2   |
| T3    | 1 | 2 |   4   |  3   |
| T4    | 0 | 1 |   1   |  4   |

We want the Shapley value of feature **A** for **T3** (A=1, B=2). With `d=2` features there are two coalitions for A: `S={}` and `S={B}`.

**Marginal for S={}** (only A is being evaluated, no other features randomised yet):

Construct `sample_size=2` composite points by drawing A-values from the other tuners (say A=3 from T1, A=2 from T2):

- `X_modded1` = T3 with A kept at 1: `[[1,2],[1,2]]` — score 4, rank 3 in both cases
- `X_modded2` = T3 with A replaced: `[[3,2],[2,2]]` — scores 8, 6, ranks 1 and 3

```
ι({}, A) = −1 · mean(rank(X_modded2) − rank(X_modded1))
         = −1 · mean([1−3, 3−3])
         = −1 · (−1.0) = +1.0
```

T3's actual A=1 is hurting its rank; the random draws had higher A values, which would have helped. The ι function records this as a negative marginal for A in this coalition, flipped to positive by the `−1` multiplier.

**Marginal for S={B}** (B is also randomised; draw B=1, B=3):

- `X_modded1` = A fixed at 1, B randomised: `[[1,1],[1,3]]` — scores 3, 5
- `X_modded2` = both A and B randomised: `[[3,1],[2,3]]` — scores 7, 7

```
ranks(X_modded1) = [4, 3]
ranks(X_modded2) = [1, 2]
ι({B}, A) = −1 · mean([1−4, 2−3]) = −1 · (−2.0) = +2.0
```

**Shapley value** (weight each marginal by `1 / [C(1,|S|) · 2]` = 0.5 each):

```
φ_A(T3) = 0.5 · 1.0 + 0.5 · 2.0 = +1.5
```

T3's feature A value (=1, below the population average) is **hurting** its rank by about 1.5 positions — it would rank roughly 1.5 places better if A were at the population average.

---

## 5. The Efficiency Property and Its Claimed Baseline

The paper claims the Shapley efficiency axiom holds for rank QoI, with:

```
Σ_j φ_j(v) = E[rank(v) | all features randomised] − rank(v)
           = base_rank − rank(v)
```

where `base_rank = (n+1)/2` is the mean ordinal rank. The paper demonstrates this in Example 3 (§6.1):

> "The sum of feature weights −20.78 − 19.52 − 18.71 − 1.95 = −60.96 captures the displacement of Texas A&M University in the ranking relative to the **middle of the ranked list (position 94.5 out of 189)**: 94.5 − 60.96 = 33.54."  
> — Pliatsika et al. (2025), §6.1

For a dataset of 189 items the middle is `(189+1)/2 = 95`, and the achieved fidelity is 0.998. The baseline `(n+1)/2` is thus the correct anchor for interpreting the sign and magnitude of the Shapley values: an item with `sum(φ) > 0` ranks better than the dataset midpoint; `sum(φ) < 0` ranks worse.

---

## 6. How the Library Implements This (and Where It Approximates)

The paper's **Algorithm 2** computes the rank of each composite point within a modified dataset `D1 = D \ {v} ∪ {u1}` and `D2 = D \ {v} ∪ {u2}`. This means for every composite point generated during the Shapley computation, the entire ranking pool is rebuilt.

The `xai-sharp` library simplifies this: `BaseRankQoI.rank()` always ranks composite points against the **fixed, unmodified** `self.X`. The item being explained is never removed from the pool, and the pool is never augmented with the composite point. This is a deliberate efficiency trade-off — rebuilding D1/D2 for every coalition of every item would be O(n² · 2^d) model evaluations.

The consequence is that efficiency holds **approximately**, not exactly. The approximation error is:

- **Small** when `n` is large: one item's presence or absence changes any other item's rank by at most 1, so the error per composite point is bounded, and its effect on the Shapley average shrinks as `1/n`.
- **Visible** when `n` is small: with 5 items the efficiency gap can be O(1) rank positions per Shapley value.

In practice, on datasets the size typically encountered here (tens to hundreds of datasets), the approximation is adequate for the relative-importance conclusions drawn from `mean_abs_shap` and the beeswarm plot. The paper's own fidelity of 0.998 on 189 items confirms this.

---

## 7. What This Means for This Codebase

### What it does not affect

- The `shap_values` matrix from `explainer.all(X)` — these are valid, consistent estimates of feature importance under the rank QoI.
- The importance ranking (`mean_abs_shap`), directional sign, and relative ordering of features — all valid.
- The beeswarm plot and bar chart — unaffected.
- The partial dependence analysis — entirely independent of the Shapley computation.

### What was changed as a result of this analysis

`SharpResults.base_value` (which stored `(n+1)/2`) and the corresponding computation in `compute_shap_values` have been **removed**. The field only existed to support a potential waterfall-style additive decomposition plot (`rank = base_rank + Σ φ_j`). Since:

1. Waterfall plots are not part of the analysis pipeline.
2. The efficiency identity holds only approximately under the library's implementation.

storing the baseline creates a misleading implied contract without serving any actual use.

---

## 8. Reading the Shapley Values Correctly

Given the sign convention (`positive φ = feature helps rank`), values should be read as:

| `φ_j(v)` | Interpretation |
|-----------|----------------|
| Large positive | Feature `j`'s value for this item is **above** the population distribution in a way that significantly improves its rank |
| Near zero | Feature `j` has little effect on this item's rank compared to a random draw |
| Large negative | Feature `j`'s value for this item is **below** the population distribution in a way that significantly worsens its rank |

`mean_abs_shap` across items gives a global importance measure: features with high `mean_abs_shap` are those whose population variation most strongly drives rank differences — regardless of direction.

---

## References

- Pliatsika, V., Fonseca, J., Akhynko, K., Shevchenko, I., & Stoyanovich, J. (2025). ShaRP: Explaining Rankings and Preferences with Shapley Values. *PVLDB*, 18(11): 4131–4143. [doi:10.14778/3749646.3749682](https://doi.org/10.14778/3749646.3749682)
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2: 307–317.
- Datta, A., Sen, S., & Zick, Y. (2016). Algorithmic transparency via quantitative input influence. *IEEE S&P*.
