import logging
logger=logging.getLogger(__name__)
import torch
import numpy as np
import seotbx

class PolsarHomogeneousPatchDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, signatures_dataset_path, patch_size=64, num_patches=1000, num_looks=1, transform=None, **kwargs):
        self.patch_size = patch_size
        self.num_patches= num_patches
        self.transform = transform
        self.num_looks = num_looks
        self.num_samples = self.patch_size*self.patch_size

        self.signatures_dataset = None
        try:
            self.signatures_dataset = np.load(signatures_dataset_path)
            logger.info(f"load file {self.signatures_dataset.shape} ({self.signatures_dataset.dtype}): {signatures_dataset_path}")
        except:
            logger.fatal(f"unable to load file: {signatures_dataset_path}")
        self.num_signatures = self.signatures_dataset.shape[0]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):

        l = np.random.uniform(low=0.01, high=2.0)
        k = np.random.randint(low=0,high=self.num_signatures)
        M_in = self.signatures_dataset[k]
        M_in_9 = seotbx.polsarproc.convert.MX3_to_X3(M_in)*l
        y9 = np.tile(M_in_9, self.num_samples ).reshape( self.patch_size, self.patch_size ,9).transpose(2,0,1)
        x9 = seotbx.polsarproc.convert.MX3_to_X3(
            seotbx.polsarproc.tools.polsar_n_looks_simulation(
                M_in,self.num_looks, self.num_samples).transpose(1,2,0)).reshape(self.patch_size,
                                                                                 self.patch_size ,9).transpose(2,0,1)*l

        return (torch.from_numpy(x9.astype("float32")), torch.from_numpy(y9.astype("float32")))