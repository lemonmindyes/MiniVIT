from dataclasses import dataclass


# tiny: 3041320. n_layer = 6, epoch = 200
# mini: 5701288. n_layer = 12, epoch = 300
@dataclass
class Config:
    img_size: int = 224
    patch_size: int = 16
    channel: int = 3
    dim: int = 192
    dropout_rate: float = 0.0
    n_class: int = 1000
    # transformer
    n_head: int = 3
    n_layer: int = 12
    mode: str = 'cls'