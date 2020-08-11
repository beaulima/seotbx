import logging
import seotbx
import numpy as np
import numpy as np
logger = logging.getLogger("seotbx.polsarproc.apps.create_polsar_samples")


def polsarsamples_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create synthetic polarimetric samples from sigs")
    ap.add_argument("sigs_dataset", type=str, help="path to the polarimetric signatures datasets")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")

def polsarsamples_application_func(args):
    """Synthetic polarimetric signatures.

    """
    signatures_dataset = None
    try:
        signatures_dataset = np.load(args.sigs_dataset)
        logger.info(f"load file {signatures_dataset.shape} ({signatures_dataset.dtype}): {args.sigs_dataset}")
    except:
        logger.fatal(f"unable to load file: {args.sigs_dataset}")

    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_dataset.transpose(1, 2, 0), False)

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha, bshow=True)
