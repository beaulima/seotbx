import numpy as np

# TENSOR_SHAPE
TS_BCHW = 0
TS_BHWC = 1
TS_CHW = 2
TS_HWC = 3
TS_HW = 4


def get_channel_value(data, shape_type):
    if shape_type == TS_BCHW:
        return data.shape[1]
    if shape_type == TS_BHWC:
        return data.shape[3]
    if shape_type == TS_CHW:
        return data.shape[0]
    if shape_type == TS_HWC:
        return data.shape[2]
    raise Exception("wrong shape type")


def len_keys(dict):
    return len(dict.keys())


POLYTYPECONF_LIST = [
    "S2", "S2C3", "S2C4", "S2T3", "S2T4",
    "S2SPPpp1", "S2SPPpp2", "S2SPPpp3", "S2IPPpp4",
    "S2IPPpp5", "S2IPPpp6", "S2IPPpp7", "S2IPPfull",
    "S2C2pp1", "S2C2pp2", "S2C2pp3",
    "S2C2lhv", "S2C2rhv", "S2C2pi4",
    "S2SPPlhv", "S2SPPrhv", "S2SPPpi4",
    "C2", "C2IPPpp5", "C2IPPpp6", "C2IPPpp7",
    "C3", "C3T3", "C3C2pp1", "C3C2pp2", "C3C2pp3",
    "C3C2lhv", "C3C2rhv", "C3C2pi4",
    "C3IPPpp4", "C3IPPpp5", "C3IPPpp6", "C3IPPpp7",
    "T3", "T3C3", "T3C2pp1", "T3C2pp2", "T3C2pp3",
    "T3C2lhv", "T3C2rhv", "T3C2pi4",
    "T3IPPpp4", "T3IPPpp5", "T3IPPpp6", "T3IPPpp7",
    "C4", "C4T4", "C4C3", "C4T3",
    "C4C2pp1", "C4C2pp2", "C4C2pp3",
    "C4C2lhv", "C4C2rhv", "C4C2pi4",
    "C4IPPpp4", "C4IPPpp5", "C4IPPpp6", "C4IPPpp7", "C4IPPfull",
    "T4", "T4C4", "T4C3", "T4T3",
    "T4C2pp1", "T4C2pp2", "T4C2pp3",
    "T4C2lhv", "T4C2rhv", "T4C2pi4",
    "T4IPPpp4", "T4IPPpp5", "T4IPPpp6", "T4IPPpp7", "T4IPPfull",
    "T6", "SPP", "SPPIPP", "SPPC2", "IPP"]

IDX = 0
KEYS = 1

# Polsar band index definition
HH = 0
HV = 1
VH = 2
VV = 3

S2_IDX_LIST = [(HH, "HH"), (VV, "VV"), (HV, "HV"), (VH, "VH")]

S2_IDX = {
    "HH": HH,
    "HV": HV,
    "VH": VH,
    "VV": VV
}

i_HH = 0
q_HH = 1
i_HV = 2
q_HV = 3
i_VH = 4
q_VH = 5
i_VV = 6
q_VV = 7

S2_SLC_IDX = {
    "i_HH": i_HH,
    "q_HH": q_HH,
    "i_HV": i_HV,
    "q_HV": q_HV,
    "i_VH": i_VH,
    "q_VH": q_VH,
    "i_VV": i_VV,
    "q_VV": q_VV
}

S2_I_Q_IDX = {
    "HH": ["i_HH", "q_HH"],
    "HV": ["i_HV", "q_HV"],
    "VH": ["i_VH", "q_VH"],
    "VV": ["i_VV", "q_VV"]
}

# Genric matrix 3x3 index definition
M11 = 0
M12_real = 1
M12_imag = 2
M13_real = 3
M13_imag = 4
M22 = 5
M23_real = 6
M23_imag = 7
M33 = 8

M3_IDX = {"M11": M11,
          "M12_real": M12_real, "M12_imag": M12_imag,
          "M13_real": M13_real, "M13_imag": M13_imag,
          "M22": M22,
          "M23_real": M23_real, "M23_imag": M23_imag,
          "M33": M33}

# Matrix coherence index definition
T11 = 0
T12_real = 1
T12_imag = 2
T13_real = 3
T13_imag = 4
T22 = 5
T23_real = 6
T23_imag = 7
T33 = 8

T3_IDX = {"T11": T11,
          "T12_real": T12_real, "T12_imag": T12_imag,
          "T13_real": T13_real, "T13_imag": T13_imag,
          "T22": T22,
          "T23_real": T23_real, "T23_imag": T23_imag,
          "T33": T33}

T3_IDX_NAME = ["T11",
          "Re{T12}", "Im{T12}",
          "Re{T13}", "Im{T13}",
          "T22",
          "Re{T23}","Im{T23}",
          "T33"]

# Matrix covariance index definition
C11 = 0
C12_real = 1
C12_imag = 2
C13_real = 3
C13_imag = 4
C22 = 5
C23_real = 6
C23_imag = 7
C33 = 8

C3_IDX = {"C11": C11,
          "C12_real": C12_real, "C12_imag": C12_imag,
          "C13_real": C13_real, "C13_imag": C13_imag,
          "C22": C22,
          "C23_real": C23_real, "C23_imag": C23_imag,
          "C33": C33}

PAULI_T3_VIZ_IDX = {
    "R": T22,
    "G": T33,
    "B": T11
}

# H-A-Alpha index definition
Entropy = 0
Anisotropy = 1
Alpha = 2
Beta = 3
Delta = 4
Gamma = 5
Lambda = 6
Alpha1 = 7
Alpha2 = 8
Alpha3 = 9
Lambda1 = 10
Lambda2 = 11
Lambda3 = 12

HAAlpha_IDX = {
    "Entropy": Entropy,
    "Anisotropy": Anisotropy,
    "Alpha": Alpha,
    "Beta": Beta,
    "Delta": Delta,
    "Gamma": Gamma,
    "Lambda": Lambda,
    "Alpha1": Alpha1,
    "Alpha2": Alpha2,
    "Alpha3": Alpha3,
    "Lambda1": Lambda1,
    "Lambda2": Lambda2,
    "Lambda3": Lambda3
}

HAAlpha_MINIDX = {
    "Entropy": Entropy,
    "Anisotropy": Anisotropy,
    "Alpha": Alpha
}

# H and alpha decision boundaries
lim_al1 = 55.
lim_al2 = 50.
lim_al3 = 48.
lim_al4 = 42.
lim_al5 = 40.
lim_H1 = 0.9
lim_H2 = 0.5
lim_H_min = 0.0
lim_H_max = 1.0
lim_A_min = 0.0
lim_A_max = 1.0
lim_al_min = 0.0
lim_al_max = 90.0

Z1 = 0
Z2 = 1
Z3 = 2
Z4 = 3
Z5 = 4
Z6 = 5
Z7 = 6
Z8 = 7
Z9 = 8

HALPHA_DIV_IDX=["Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8","Z9"]

HALPHA_CLASS_DEF = {
    "Z1": {"id": Z1, "id_pos": (0.01, 86.0), "color": (1.0, 0.0, 0.0),
           "h_lims": (lim_H_min, lim_H2), "al_lims": (lim_al3, lim_al_max), "en_id": 'Dihedral Reflector'},
    "Z2": {"id": Z2, "id_pos": (0.01, 43.5), "color": (1.0, 0.0, 1.0),
           "h_lims": (lim_H_min, lim_H2), "al_lims": (lim_al4, lim_al3),
           "en_id": 'Dipole'},
    "Z3": {"id": Z3, "id_pos": (0.01, 1.00), "color": (0.0, 0.0, 1.0),
           "h_lims": (lim_H_min, lim_H2), "al_lims": (lim_al_min, lim_al4), "en_id": 'Bragg Surface'},
    "Z4": {"id": Z4, "id_pos": (0.51, 86.0), "color": (0.0, 1.0, 0.0),
           "h_lims": (lim_H2, lim_H1), "al_lims": (lim_al2, lim_al_max),
           "en_id": 'Double Reflection'},
    "Z5": {"id": Z5, "id_pos": (0.51, 43.5), "color": (1.0, 0.5, 0.0),
           "h_lims": (lim_H2, lim_H1), "al_lims": (lim_al5, lim_al2),
           "en_id": 'Anisotropic Particles'},
    "Z6": {"id": Z6, "id_pos": (0.51, 1.00), "color": (0.0, 1.0, 1.0),
           "h_lims": (lim_H2, lim_H1), "al_lims": (lim_al_min, lim_al5),
           "en_id": 'Random Surface'},
    "Z7": {"id": Z7, "id_pos": (0.95, 86.0), "color": (0.5, 0.5, 0.5),
           "h_lims": (lim_H1, lim_H_max), "al_lims": (lim_al1, lim_al_max), "en_id": 'Complex Structures'},
    "Z8": {"id": Z8, "id_pos": (0.95, 41.0), "color": (1.0, 1.0, 0.0),
           "h_lims": (lim_H1, lim_H_max), "al_lims": (lim_al5, lim_al1),
           "en_id": 'Random Anisotropic Scatterers'},
    "Z9": {"id": Z9, "id_pos": (0.95, 1.00), "color": (1.0, 1.0, 0.0),
           "h_lims": (lim_H1, lim_H_max), "al_lims": (lim_al_min, lim_al5), "en_id": 'Not-Feasible'}
}

OMITCLASS = "Z9"

import numpy as np

C1_SIG = np.array([5.56, -0.03, 0.36, 0.47, 0.24, 6.64, 0.24, 0.20, 4.53])
C1_HAALPHA = [0.98, 0.14, 56.6]
C9_SIG = np.array([5.40, -1.14, -0.34, 0.27, -0.33, 0.56, -0.01, 0.09, 0.16])
C9_HAALPHA = [0.26, 0.40, 18.3]
