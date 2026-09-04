# The mathematical model

This is the full formulation of the linear program that `energyplan` builds.
It is a multi-period, multi-region capacity-expansion model with storage,
transmission and policy constraints, solved on a reduced set of representative
days.

## Sets

| Symbol | Meaning |
|---|---|
| $y \in Y$ | milestone planning years, e.g. 2025, 2030, … |
| $r \in R$ | regions (demand centres) |
| $g \in G$ | generation technologies |
| $s \in S$ | storage technologies |
| $l \in L$ | transmission lines |
| $d \in D$ | representative days |
| $h \in H$ | chronological steps within a day |

Each milestone year $y$ stands for $n_y$ calendar years (the gap to the next
milestone) and is discounted by $\delta_y = (1+\rho)^{-(y-y_0)}$.
Each representative day carries a weight $w_d$ in days per year; one modelled
step lasts $\tau = 24/|H|$ hours, so the hours of the year a single step
represents is $\omega_d = w_d \tau$, and $\sum_{d,h} \omega_d = 8760$.

## Variables

All are continuous and non-negative.

| Variable | Meaning |
|---|---|
| $b_{g,r,y}$ | new capacity of $g$ commissioned in year $y$ (MW) |
| $k_{g,r,y}$ | total installed capacity in service (MW) |
| $\hat{k}_{g,r,y}$ | of which is new-build still inside its economic life (MW) |
| $x_{g,r,y,d,h}$ | generation (MW) |
| $z_{g,r,y}$ | early retirement of existing capacity (MW), optional |
| $p_{s,r,y},\; e_{s,r,y}$ | storage power (MW) and energy (MWh) capacity |
| $c^{+}_{s,r,y,d,h},\; c^{-}_{s,r,y,d,h}$ | charging and discharging (MW) |
| $q_{s,r,y,d,h}$ | state of charge (MWh) |
| $f^{\rightarrow}_{l,y,d,h},\; f^{\leftarrow}_{l,y,d,h}$ | directional line flows (MW) |
| $m_{l,y}$ | line transfer capability (MW) |
| $u_{r,y,d,h}$ | unserved energy (MW) |
| $\sigma_{r,y},\; \epsilon_y,\; \phi_{\cdot,y}$ | penalised slacks on reserve, emission cap and share targets |

## Objective

Minimise the discounted present value of total system cost:

$$
\min \;
\underbrace{\sum_{g,r,y} A_{g,y}\, b_{g,r,y} \!\!\sum_{\substack{y' \in Y \\ y \le y' < y + L_g}} \!\! \delta_{y'} n_{y'}}_{\text{capital}}
\;+\;
\sum_{y} \delta_y n_y \Big[
\sum_{g,r} F_g k_{g,r,y}
+ \sum_{g,r,d,h} \omega_d\, v_{g,y}\, x_{g,r,y,d,h}
+ \sum_{r,d,h} \omega_d\, V\, u_{r,y,d,h}
\Big]
$$

plus the analogous storage and transmission terms and the slack penalties.

* $A_{g,y} = \text{capex}_{g,y} \cdot \mathrm{CRF}(\rho_g, L_g)$ is the annuity of a
  unit built in year $y$, where $\mathrm{CRF}(\rho, L) = \rho / (1 - (1+\rho)^{-L})$.
* $v_{g,y} = \text{VOM}_g + \dfrac{\text{fuel price}_{y}}{\eta_g}
  + \dfrac{\text{CO}_2\text{ price}_y \cdot \text{emission factor}_g}{\eta_g}$
  is the short-run marginal cost.
* $V$ is the value of lost load.

**Why an annuity and not a lump sum.** Charging the full overnight cost in the
build year makes anything built near the end of the horizon look uneconomic,
because the model cannot see the service it provides afterwards. Spreading the
cost as a level annuity over the asset's life, and charging it only in the
modelled years the asset is actually in service, removes that end effect.
Charging it on the *build* variable rather than on installed capacity is what
lets capex follow a learning curve: a plant built in 2045 pays the 2045 price
for its whole life.

## Constraints

**Capacity accounting.** New capacity stays in service for its lifetime $L_g$;
existing capacity follows its own retirement schedule $K^{0}_{g,r,y}$.

$$\hat{k}_{g,r,y} = \sum_{\substack{y' \le y \\ y - y' < L_g}} b_{g,r,y'}
\qquad
k_{g,r,y} = K^{0}_{g,r,y} + \hat{k}_{g,r,y} - \sum_{y' \le y} z_{g,r,y'}$$

**Energy balance** (one per region, year and step). Its dual is the marginal
cost of electricity in that hour.

$$\sum_g x_{g,r,y,d,h}
+ \sum_s \big(c^{-}_{s,r,y,d,h} - c^{+}_{s,r,y,d,h}\big)
+ \sum_{l \to r} (1-\lambda_l) f_{l,y,d,h}
- \sum_{l \leftarrow r} f_{l,y,d,h}
+ u_{r,y,d,h}
= D_{r,y,d,h}$$

**Dispatch limit.** $\alpha$ is a fixed availability for thermal plant and an
hourly capacity factor for variable renewables; the slack is curtailment.

$$x_{g,r,y,d,h} \le \alpha_{g,d,h}\, k_{g,r,y}$$

**Annual energy budget** (hydro inflow, fuel contracts):
$\sum_{d,h} \omega_d\, x_{g,r,y,d,h} \le \mathrm{CF}^{\max}_g \cdot 8760 \cdot k_{g,r,y}$.

**Ramping** between consecutive steps of a day:
$|x_{g,r,y,d,h} - x_{g,r,y,d,h-1}| \le \gamma_g\, k_{g,r,y}$.

**Storage.** The state of charge is cyclic within each representative day, so a
day neither starts nor ends with free energy:

$$q_{s,r,y,d,h} = (1-\theta_s)^{\tau} q_{s,r,y,d,h-1}
+ \eta^{+}_s \tau\, c^{+}_{s,r,y,d,h}
- \frac{\tau}{\eta^{-}_s} c^{-}_{s,r,y,d,h},
\qquad h-1 \text{ taken modulo } |H|$$

with $q \le e$, $c^{\pm} \le p$, and duration bounds
$\underline{T}_s p \le e \le \overline{T}_s p$.

Summing the cyclic balance around a day gives
$\sum_h c^{-} = \eta^{+}\eta^{-} \sum_h c^{+}$: storage returns strictly less
energy than it absorbs, and can never be a net source. `tests/test_model.py`
asserts this directly.

**Reserve margin.** Firm capacity must cover the peak with a margin. Variable
renewables enter at their capacity credit, not their nameplate.

$$\sum_g \kappa_g k_{g,r,y} + \sum_s \kappa_s p_{s,r,y}
+ \sum_{l \sim r} 0.75(1-\lambda_l) m_{l,y} + \sigma_{r,y}
\ge P_{r,y}(1+\mu)$$

The peak $P_{r,y}$ is taken from the **full 8760-hour** series, not from the
reduced days, so clustering cannot quietly shrink the capacity requirement.

**Emissions**, per year and optionally cumulative:

$$\sum_{g,r,d,h} \omega_d \frac{\text{ef}_g}{\eta_g} x_{g,r,y,d,h} - \epsilon_y \le C_y$$

**Renewable / clean energy share**, as a fraction of demand served:

$$\sum_{g \in G^{\text{RE}}} \sum_{r,d,h} \omega_d\, x_{g,r,y,d,h} + \phi_y
\;\ge\; \pi_y \sum_r D_{r,y}$$

**Build limits.** $b_{g,r,y} \le \dot{B}_g n_y$ (annual build rate times the
period length), $b_{g,r,y} = 0$ before $y_0 + \text{lead time}_g$, and
$\sum_r k_{g,r,y} \le K^{\max}_g$ (system-wide resource potential).

## Slacks and why they exist

$\sigma$, $\epsilon$ and $\phi$ carry large backstop prices. Without them a
scenario whose targets cannot be met returns "infeasible", which tells a
planner nothing. With them the model always returns a plan and states exactly
which target it missed and by how much. Set the backstop high enough and the
constraint is effectively hard; the shadow price of a binding target is capped
at its backstop.

## Duals as prices

Because every row is named, the solver's dual values can be read back as
prices. They arrive in objective units, i.e. already multiplied by
$\delta_y n_y$ (and by $\omega_d$ for hourly rows), so `energyplan` divides
that weight out before reporting:

| Row | Dual, once rescaled |
|---|---|
| energy balance | marginal cost of electricity, USD/MWh |
| reserve margin | marginal value of firm capacity, USD/MW-year |
| emission cap | shadow carbon price, USD/tCO₂ |

**A caveat on degeneracy.** At an emission cap of exactly zero the optimum sits
on a degenerate vertex: several dual solutions are valid and the solver may
return the backstop price rather than the true marginal abatement cost.
`energyplan` detects this and marks the value with `≥`.
`PlanResult.empirical_marginal_carbon_cost(year)` re-solves with a slightly
relaxed cap and returns the finite difference, which is well defined either way.

## Time-domain reduction

Days are clustered with k-medoids on the joint demand-and-renewables feature
vector. Medoids rather than centroids, so every representative day is a *real*
day whose internal chronology is physically consistent. The day containing the
highest net load is pinned as a medoid and always survives the reduction.
Cluster weights are rescaled to reproduce exactly 365 days, and each reduced
profile is scaled so its weighted annual mean matches the original 8760-hour
series.

## What the model does not do

Being a linear program, it has no integer variables, so it cannot represent
unit commitment, minimum up and down times, start-up costs, or lumpy plant
sizes. Its dispatch is therefore optimistic relative to an hourly unit-commitment
model, and the value of flexibility is understated. It is also deterministic
and has perfect foresight: it knows every future fuel price and weather year in
advance, so it under-states the option value of flexible assets. Both are the
standard trade-offs of long-run capacity planning; treat the output as a
screening result to be tested, not as an operating schedule.
