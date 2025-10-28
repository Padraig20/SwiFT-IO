from .encoder.swin4d_transformer_ver7 import SwinTransformer4D
from .encoder.swin4d_transformer_ver9 import SwinTransformer4D
from .encoder.lstm_encoder import LSTMEncoder, LSTMEncoderLight
from .decoder.single_target_decoder import SingleTargetDecoder
from .decoder.series_decoder import SeriesDecoder
from .decoder.lstm_decoder import LSTMRegressionHead, LSTMSeriesRegressionHead

def load_model(model_name, hparams=None):

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True
        
    h, w, d, t_orig = hparams.img_size
    hp, wp, dp, tp = hparams.patch_size
    
    h = h // (hp*8) if hp != 1 else h
    w = w // (wp*8) if wp != 1 else w
    d = d // (dp*8) if dp != 1 else d
    t = t_orig // tp if tp != 1 else t_orig

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
    elif model_name == "swin4d_ver9":
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
    elif model_name == "single_target_decoder": # TODO add hparams?
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
    elif model_name == "series_decoder":
        num_classes = 1 if hparams.downstream_task_type == 'regression' else hparams.num_classes
        net = SeriesDecoder(
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
            num_output_queries=t_orig,
            #num_output_query_channels=hparams.num_output_query_channels,
            num_classes=num_classes,
            num_targets=hparams.num_targets,
            downstream_task_type=hparams.downstream_task_type
        )
    elif model_name == "lstm_encoder":
        net = LSTMEncoder(
            hidden_dim=getattr(hparams, 'lstm_hidden_dim', 256),
            num_layers=getattr(hparams, 'lstm_num_layers', 2),
            dropout=getattr(hparams, 'lstm_dropout', 0.3),
            pooling_type=getattr(hparams, 'lstm_pooling', 'adaptive'),
            pooled_spatial_dim=getattr(hparams, 'lstm_pooled_dim', 16),
            bidirectional=getattr(hparams, 'lstm_bidirectional', False),
            return_sequence=False
        )
    elif model_name == "lstm_encoder_light":
        net = LSTMEncoderLight(
            hidden_dim=getattr(hparams, 'lstm_hidden_dim', 128),
            num_layers=getattr(hparams, 'lstm_num_layers', 2),
            dropout=getattr(hparams, 'lstm_dropout', 0.3),
            pooled_spatial_dim=getattr(hparams, 'lstm_pooled_dim', 8)
        )
    elif model_name == "lstm_regression_head":
        # Get LSTM encoder output dim
        lstm_hidden = getattr(hparams, 'lstm_hidden_dim', 256)
        lstm_bidirectional = getattr(hparams, 'lstm_bidirectional', False)
        input_dim = lstm_hidden * (2 if lstm_bidirectional else 1)

        net = LSTMRegressionHead(
            input_dim=input_dim,
            num_targets=hparams.num_targets,
            hidden_dim=getattr(hparams, 'lstm_decoder_hidden', 128),
            dropout=getattr(hparams, 'lstm_dropout', 0.3)
        )
    elif model_name == "lstm_series_regression_head":
        # Get LSTM encoder output dim
        lstm_hidden = getattr(hparams, 'lstm_hidden_dim', 256)
        lstm_bidirectional = getattr(hparams, 'lstm_bidirectional', False)
        input_dim = lstm_hidden * (2 if lstm_bidirectional else 1)

        net = LSTMSeriesRegressionHead(
            input_dim=input_dim,
            num_timepoints=hparams.img_size[3],  # time dimension
            num_targets=hparams.num_targets,
            hidden_dim=getattr(hparams, 'lstm_decoder_hidden', 128),
            dropout=getattr(hparams, 'lstm_dropout', 0.3)
        )
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net