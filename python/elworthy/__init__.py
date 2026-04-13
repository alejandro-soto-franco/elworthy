"""elworthy: Bismut-Elworthy-Li Malliavin weights for Monte Carlo Greeks.

Two-tier API:

- Low-level (power users): ``bel_weights_constant_flow``,
  ``bel_weights_tangent_flow`` return NumPy arrays of per-path weights.
  Multiply by any vectorised payoff in NumPy / PyTorch / JAX::

      w = elworthy.bel_weights_constant_flow(w_terminals, T=1.0, sigma_at_x0=20.0)
      delta = (np.maximum(X_T - K, 0.0) * w).mean()

- High-level (defaults): ``price_and_delta_constant_flow`` and
  ``price_and_delta_tangent_flow`` accept a built-in payoff name and
  return ``(price, delta, price_stderr, delta_stderr)``.
"""

from ._elworthy import (
    __version__,
    bel_weights_constant_flow,
    bel_weights_tangent_flow,
    bel_weights_tangent_flow_parallel,
    price_and_delta_constant_flow,
    price_and_delta_tangent_flow,
    price_from_samples,
)

__all__ = [
    "__version__",
    "bel_weights_constant_flow",
    "bel_weights_tangent_flow",
    "bel_weights_tangent_flow_parallel",
    "price_and_delta_constant_flow",
    "price_and_delta_tangent_flow",
    "price_from_samples",
]
