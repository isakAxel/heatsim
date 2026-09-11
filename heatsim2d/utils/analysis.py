import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import heatsim2d as hs2
import pandas as pd



def c_analysis(n_dU,T_end):

    coeffs = hs2.compute_coeffs(n_dU,T_end,hs2.par)
    df = pd.DataFrame(data=coeffs)

    
    print("------ Relative Coefficient values --------")
    print(df.to_string())

        


n = 6
T_end = 200
c_analysis(n,T_end)