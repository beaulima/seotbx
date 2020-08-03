import logging

logger = logging.getLogger("seotbx.snappy_api.sentinel1.utils")
import seotbx
import snappy
from snappy import ProductIO


def run_gpf(gpf_process_id, parameters, product_input):
        return snappy.GPF.createProduct(gpf_process_id, parameters, product_input)

def apply_orbit_file(product_input,
                     continue_on_fail=False,
                     orbit_type='Sentinel Precise (Auto Download)',
                     poly_degree=3):
    HashMap = snappy.jpy.get_type('java.util.HashMap')
    gpf_process_id = "Apply-Orbit-File"
    parameters = HashMap()
    parameters.put('continueOnFail', continue_on_fail)
    parameters.put('orbitType', orbit_type)
    parameters.put('polyDegree=', poly_degree)
    return run_gpf(gpf_process_id, parameters, product_input)


def remove_grd_border_noise(product_input,
                     selected_polarisations="",
                     border_limit=500,
                     trim_threshold=0.5):
    "not working"
    HashMap = snappy.jpy.get_type('java.util.HashMap')
    gpf_process_id = "Remove-GRD-Border-Noise"
    parameters = HashMap()
    parameters.put('selectedPolarisations', selected_polarisations)
    parameters.put('borderLimit', border_limit)
    parameters.put('trimThreshold', trim_threshold)
    return run_gpf(gpf_process_id, parameters, product_input)


def thermal_noise_removal(product_input,
                     selected_polarisations="",
                     remove_thermal_noise=True,
                     reintroduce_thermal_noise=False):
    HashMap = snappy.jpy.get_type('java.util.HashMap')
    gpf_process_id = "ThermalNoiseRemoval"
    parameters = HashMap()
    parameters.put('selectedPolarisations', selected_polarisations)
    parameters.put('removeThermalNoise', remove_thermal_noise)
    parameters.put('reIntroduceThermalNoise', reintroduce_thermal_noise)
    return run_gpf(gpf_process_id, parameters, product_input)


def calibration(product_input,
                aux_file="Latest Auxiliary File",
                external_auxfile="",
                output_image_in_complex=True,
                output_image_scale_in_db=False,
                create_gamma_band=False,
                create_beta_band=False,
                selected_polarisations="VH,VV",
                output_sigma_band=True,
                output_gamma_band=False,
                output_beta_band=False,
                output_dn_band=False
                ):

    HashMap = snappy.jpy.get_type('java.util.HashMap')
    gpf_process_id = "Calibration"
    parameters = HashMap()
    parameters.put('auxFile', aux_file)
    parameters.put('externalAuxFile', external_auxfile)
    parameters.put('outputImageInComplex', output_image_in_complex)
    parameters.put('outputImageScaleInDb', output_image_scale_in_db)
    parameters.put('createGammaBand', create_gamma_band)
    parameters.put('createBetaBand', create_beta_band)
    parameters.put('selectedPolarisations', selected_polarisations)
    parameters.put('outputSigmaBand', output_sigma_band)
    parameters.put('outputGammaBand', output_gamma_band)
    parameters.put('outputBetaBand', output_beta_band)
    parameters.put('outputDNBand', output_dn_band)
    return run_gpf(gpf_process_id, parameters, product_input)

def topsar_deburst_SLC(product_input,
                       selected_polarisations="VH,VV"):
    HashMap = snappy.jpy.get_type('java.util.HashMap')
    gpf_process_id = "TOPSAR-Deburst"
    parameters = HashMap()
    parameters.put('selectedPolarisations', selected_polarisations)
    return run_gpf(gpf_process_id, parameters, product_input)