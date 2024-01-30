import pandas as pd
import numpy as np
import pytest
import os
from AutoCleanse.utils import *

@pytest.fixture
def data_fixture():
    data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}).astype(float)
    return data

@pytest.mark.utils
def test_replace_with_nan(data_fixture):
    expected_output = pd.DataFrame({'A': [1, 2, 3, 4, np.nan], 'B': [np.nan, 7, 8, 9, 10]}).astype(float)
    output = replace_with_nan(data_fixture, 0.2, 42)
    assert expected_output.equals(output)
