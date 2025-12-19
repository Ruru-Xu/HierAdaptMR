from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from data_loading.subsample import CmrxRecon25MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(
        data: torch.Tensor,
        mask_func: CmrxRecon25MaskFunc,
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Optional[Sequence[int]] = None,
        slice_idx: Optional[int] = None,
        num_t: Optional[int] = None,
        num_slc: Optional[int] = None,
        mask_type: Optional[str] = None,
        acc: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.
    """
    # Validate input
    if mask_func is None:
        raise ValueError("mask_func cannot be None")

    if not isinstance(mask_func, CmrxRecon25MaskFunc):
        raise ValueError(f"Expected CmrxRecon25MaskFunc, got {type(mask_func)}")

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])

    # Call the mask function
    if num_t is not None:
        mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed, slice_idx, num_t, num_slc, mask_type, acc)
    else:
        mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed, mask_type, acc)

    # Apply padding
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1]:, :] = 0

    # Repeat mask for multiple coils if needed
    if mask.shape[0] != 1:
        mask = mask.repeat_interleave(data.shape[0] // mask.shape[0], dim=0)

    # Apply mask
    masked_data = data * mask + 0.0

    return masked_data, mask, num_low_frequencies


class PromptMRSample(NamedTuple):
    """A sample of masked k-space for variational network reconstruction."""
    fully_kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    mask_type: str
    num_t: int
    num_slc: int


class CmrxReconDataTransform:
    """CmrxRecon25 Data Transformer for training"""

    def __init__(self,
                 mask_func: Optional[CmrxRecon25MaskFunc] = None,
                 uniform_resolution=None,
                 use_seed: bool = True,
                 mask_type: Optional[str] = None,
                 test_num_low_frequencies: Optional[int] = None):
        """
        Args:
            mask_func: Optional; A function that can create a mask of appropriate shape.
            use_seed: If True, computes a pseudo random number generator seed from filename.
        """
        if mask_func is None and mask_type is None:
            raise ValueError("Either `mask_func` or `mask_type` must be specified.")
        if mask_func is not None and mask_type is not None:
            raise ValueError("Both `mask_func` and `mask_type` cannot be set at the same time.")

        # Validate mask_func type if provided
        if mask_func is not None and not isinstance(mask_func, CmrxRecon25MaskFunc):
            raise ValueError(f"mask_func must be CmrxRecon25MaskFunc, got {type(mask_func)}")

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_resolution = uniform_resolution

        if mask_func is None:
            self.mask_type = mask_type
            self.num_low_frequencies = test_num_low_frequencies

    def __call__(self,
                 kspace: np.ndarray,
                 mask: np.ndarray,
                 target: np.ndarray,
                 attrs: Dict,
                 fname: str,
                 slice_num: int,
                 num_t: int,
                 num_slc: int,
                 mask_type: str,
                 acc: int
                 ) -> PromptMRSample:
        """Process k-space data and return PromptMRSample."""

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end),
                slice_idx=slice_num, num_t=num_t, num_slc=num_slc, mask_type=mask_type, acc=acc
            )
        else:
            masked_kspace = kspace_torch
            mask_torch = to_tensor(mask)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0
            if 'ktRadial' in fname:
                mask_type = 'kt_radial'
            elif 'ktGaussian' in fname:
                mask_type = 'kt_gaussian'
            elif 'Uniform' in fname:
                mask_type = 'uniform'
            num_low_frequencies = self.num_low_frequencies

        sample = PromptMRSample(
            fully_kspace=kspace_torch,
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            mask_type=mask_type,
            num_t=num_t,
            num_slc=num_slc,
        )

        return sample


