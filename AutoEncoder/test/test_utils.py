import pandas as pd
import numpy as np
import unittest
import os
from AutoEncoder.utils import *

data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}).astype(float)

class test_utils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_utils, self).__init__(*args, **kwargs)
        self.df = data

    def test_replace_with_nan(self):
        output = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [np.nan, np.nan, 8, 9, 10]}).astype(float)
        df_o = replace_with_nan(self.df,0.2,42)
        assert output.equals(df_o)

if __name__ == "__main__":
    unittest.main()
