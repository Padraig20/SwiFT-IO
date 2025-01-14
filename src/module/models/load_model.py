from .SwiFT.swin4d_transformer_ver7 import SwinTransformer4D
from .PerceiverIO.single_target_decoder import SingleTargetDecoder

def load_model(model_name, hparams=None):

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True
        
    h, w, d, t = hparams.img_size
    hp, wp, dp, tp = hparams.patch_size
        
    h = h // hp // hp // hp
    w = w // wp // hp // hp
    d = d // dp // dp // dp
    t = t // tp // tp // tp
    embed_dim = hparams.embed_dim * 8
    
    dims = h * w * d * t
        
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
        )
    elif model_name == "single_target_decoder": # TODO add hparams
        num_classes = 1 if hparams.downstream_task_type == 'regression' else hparams.num_classes
        net = SingleTargetDecoder(
            num_latents=embed_dim,
            num_latent_channels=dims, # TODO: verify this
            #activation_checkpointing=hparams.activation_checkpointing,
            #activation_offloading=hparams.activation_offloading,
            #num_cross_attention_heads=hparams.num_cross_attention_heads,
            #num_cross_attention_qk_channels=hparams.num_cross_attention_qk_channels,
            #num_cross_attention_v_channels=hparams.num_cross_attention_v_channels,
            #cross_attention_widening_factor=hparams.cross_attention_widening_factor,
            #cross_attention_residual=hparams.cross_attention_residual,
            #dropout=hparams.dropout,
            #init_scale=hparams.init_scale,
            #freeze=hparams.freeze,
            #num_output_queries=hparams.num_output_queries,
            #num_output_query_channels=hparams.num_output_query_channels,
            num_classes=num_classes
        )
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net