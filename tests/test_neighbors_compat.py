import numpy as np
import pytest
from cace.data.neighborhood import get_neighborhood, get_neighborhood_ASE


@pytest.mark.parametrize('pbc', [(True,True,True),(False,False,False),(True,False,True)])
def test_equivalent_neighbor_shift_sets(pbc):
    positions = np.array([[.2,.3,.4],[2.8,.4,.8],[1.2,1.4,1.8]])
    cell = np.diag([3.,4.,5.])
    routes = [f(positions, 2.1, pbc=pbc, cell=cell.copy(), true_self_interaction=False)
              for f in (get_neighborhood, get_neighborhood_ASE)]
    rows = [sorted(tuple(row) for row in np.column_stack((edge.T, units, shifts)))
            for edge, shifts, units in routes]
    np.testing.assert_allclose(rows[0], rows[1])
