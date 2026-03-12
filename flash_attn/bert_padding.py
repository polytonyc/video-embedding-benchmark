"""Stub — unpad_input/pad_input are never called when use_flash_attn=False."""


def unpad_input(*args, **kwargs):
    raise NotImplementedError("flash_attn not available — use_flash_attn must be False")


def pad_input(*args, **kwargs):
    raise NotImplementedError("flash_attn not available — use_flash_attn must be False")
