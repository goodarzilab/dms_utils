import os
import sys
import numpy as np


current_script_path = sys.argv[0]
package_home_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
if package_home_path not in sys.path:
    sys.path.append(package_home_path)

from dms_utils.utils import utils as dms_utils

def test_filter_by_distance_between_mutations():
    a = np.zeros((5, 6))
    a[0, 0] = 1
    a[0, 4] = 1
    a[1, 0] = 1
    a[1, 3] = 1
    a[2, 2] = 1
    a[3, 1] = 1
    a[3, 3] = 1
    a[4, 4] = 1
    a[4, 5] = 1

    where_too_close = dms_utils.filter_by_distance_between_mutations(a, max_prohib_distance = 3)
    assert (where_too_close == np.array([0,1,0,1,1], dtype=np.bool)).all()


def test_filter_by_positions_surrounding_mutations():
    a = np.zeros((5, 6))
    a[0, 0] = 1
    a[0, 1] = -1
    a[2, 1] = -1
    a[2, 2] = 1
    a[3, 1] = 1
    a[3, 3] = -1

    where_too_close = dms_utils.filter_by_positions_surrounding_mutations(a)
    assert (where_too_close == np.array([1,0,1,0,0], dtype=np.bool)).all()


if __name__ == "__main__":
    test_filter_by_distance_between_mutations()
    test_filter_by_positions_surrounding_mutations()