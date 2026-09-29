import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import Sim.heatsim2d as s
import pandas as pd



def c_analysis(n_dU,T_end):

    coeffs = s.compute_coeffs(n_dU,T_end,s.par)
    df = pd.DataFrame(data=coeffs)

    
    print("------ Relative Coefficient values --------")
    print(df.to_string())

        


n = 11
T_end = 1000
c_analysis(n,T_end)