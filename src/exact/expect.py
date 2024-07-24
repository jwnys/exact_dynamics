import jax
from netket.vqs.base import VariationalState, expect#, expect_and_forces
from typing import Any, Callable, Optional
from netket.utils.types import PyTree, SeedT, NNInitFunc, Array, DType, Union
from netket.operator import DiscreteOperator
from netket.stats import Stats
import jax.numpy as jnp
from netket.vqs.full_summ.expect import _check_hilbert
from .utils import to_matrix
from .state import NonVariationalVectorState, _normalize


@expect.dispatch
def expect(vstate: NonVariationalVectorState, O: Union[DiscreteOperator, Array]) -> Stats:  # noqa: F811
    _check_hilbert(vstate, O)

    if isinstance(O, DiscreteOperator):    
        O = to_matrix(O, cache=True)

    return _compute_expval(O, vstate.normalized_vector)

# @jax.jit
def _compute_expval(O, psi):
    psi = _normalize(psi)
    Opsi = O @ psi
    expval_O = (psi.conj().dot(Opsi)).sum()
    variance = jnp.sum(jnp.abs(Opsi - expval_O * psi) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)
