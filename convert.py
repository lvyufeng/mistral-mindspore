try:
    import torch
except Exception as exc:
    raise ImportError("'import torch' failed, please install torch by "
                    "`pip install torch` or instructions from 'https://pytorch.org'") \
                    from exc
import os
import logging
import numpy as np
import fire
import mindspore
from mindspore import Tensor

def convert_torch_to_mindspore(pth_file):
    """convert torch checkpoint to mindspore"""
    ms_ckpt_path = pth_file.replace('.pth', '.ckpt')

    state_dict = torch.load(pth_file, map_location='cpu')

    if os.path.exists(ms_ckpt_path):
        return ms_ckpt_path

    ms_ckpt = []
    logging.info('Starting checkpoint conversion.')

    has_bf16 = False
    for key, value in state_dict.items():
        if value.dtype == torch.bfloat16:
            data = Tensor(value.to(torch.float).numpy(), dtype=mindspore.float16)
            if not has_bf16:
                has_bf16 = True
        else:
            data = Tensor(value.numpy())
        ms_ckpt.append({'name': key, 'data': data})

    if has_bf16:
        logging.warning("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")

    try:
        mindspore.save_checkpoint(ms_ckpt, ms_ckpt_path)
    except Exception as exc:
        raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, '
                            f'please checkout the path.') from exc

    return ms_ckpt_path
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(convert_torch_to_mindspore)
