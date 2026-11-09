"""E1.8 Lagrangian variant: the price of a violation is a multiplier lambda >= 0, updated once per rollout.

    lambda <- max(0, lambda + lr * (violations per step - target)),  target = 0

This is the plain integral update of Stooke, Achiam and Abbeel (2020, Eq. 22), with the cost measured as
violations per step of the last rollout instead of the mean episode cost, and the multiplier used as a
price in the reward instead of a separate cost critic. Defaults (see train_agent.py): lambda_0 = 1, as in
Ray et al. (2019, safety-starter-agents penalty_init = 1) and Stooke et al. (Section 7); lr = 3, from
Stooke et al.'s K_I = 1e-2 on the episode cost, converted to violations per step with our 300-step limit.
"""


def lagrange_update(lam: float, lr: float, violations: int, steps: int, target: float = 0.0) -> float:
    if steps <= 0:
        return lam
    return max(0.0, lam + lr * (violations / steps - target))
