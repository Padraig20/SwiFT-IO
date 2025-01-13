from .SwiFT.swin4d_transformer_ver7 import SwinTransformer4D

def load_model(model_name, hparams=None):
    #number of transformer stages
    #n_stages = len(hparams.depths) -> assume four stages

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    print(to_float)
    
    h, w, d, t = hparams.img_size
    dims = h//48 * w//48 * d//48 * hparams.embed_dim*8 # TODO: verify this
    
    if model_name == "swin4d_ver7":
        net = SwinTransformer4D(
            img_size=hparams.img_size,
            in_chans=hparams.in_chans,
            embed_dim=hparams.embed_dim,
            window_size=hparams.window_size,
            first_window_size=hparams.first_window_size,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            num_heads=hparams.num_heads,
            c_multiplier=hparams.c_multiplier,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float = to_float,
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate
        ) # TODO add decoder
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net