# import numpy as np
from star import ProtonSTARCalculator,ProtonMaterials
from mendeleev import element


"""
    Module where different types of utilities will reside, such as data-processing, fetching
    or other parts of the pipeline which is not the actual simulation
"""

### Material parameters are

def import_PSTAR_data(material,Ep):
    """
    Help-function to import PSTAR data 
    #TODO: #FIXME

    Params: 
        material(str):
            Material denoted by either the Element Symbol or the shortened alloy-name
        Ep(float):
            Proton-energy in MeV
        param(idk):
            idk

    Returns:
        dS(float):
            Stopping power of target material, at interpolated values param

    """
    def get_proton_material(mat :str, protonMaterial):
        """
        Converts Element symbol to full name to ProtonMaterials enum object
        for example "Al" -> "ALUMINUM" -> ProtonMaterials.ALUMINUM
        """
        PSTARName = element(mat).name.upper()
        return getattr(protonMaterial,PSTARName)
    
    mat = get_proton_material(material,ProtonMaterials)
    calc = ProtonSTARCalculator(mat)
    dS = calc.calculate_total_stopping_powers(Ep)

    return dS



def import_thermal_data(material):


    if material in SOLIDS_TABLE:
        mat_entry = SOLIDS_TABLE[material]
        rho = mat_entry["rho"]
        c_p = mat_entry["cp"]
        k = mat_entry["k"]
        return k,rho,c_p        

    ### Mendeleev data
    el = element(material)
    k = el.thermal_conductivity  # [W/m K]
    k = k / 100          # 1m = 100cm [W/cm K]
    rho = el.density             # [g/cm**3]
    c_p = el.specific_heat_capacity # spec heat @ 25 C, 1bar [J/g/K]

    return k,rho,c_p

def import_mat_data(material):

    k,rho,c_p =import_thermal_data(material)
    eps = EMISSIVITY_TABLE[material]
    return k,rho,c_p,eps
    

EMISSIVITY_TABLE = {
    "Al" : 0.2,
    "Ni" : 0.4,
    "SS316L" : 0.23,
}

SOLIDS_TABLE = {
    # units in [g/cm3], [J/gK], [W/mK]
    "SS316L" : {"rho": 8, "cp": 0.5, "k" : 16.3 },
}

if __name__ == '__main__':
    k,rho,cp,eps = import_mat_data("SS316L")
    print(k,rho,cp)
    print("epsy",eps)
    # Ep = np.linspace(500,1000,11)
    Ep = 800
    dS = import_PSTAR_data("Al",Ep)
    print(f"dS : {dS}")