import torch
import re


def np_collate(batch):
    r"""Keep all numpy tensors as np tensors, without casting to torch."""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if re.search("[SaUO]", elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            # We return the batch of np tensors as is, without converting to torch
            return [b for b in batch]
        if elem.shape == ():  # scalars
            # We return the batch of np tensors as is, without converting to torch
            return [batch]

    import ipdb; ipdb.set_trace()
    raise TypeError((error_msg.format(type(batch[0]))))
