import torch
from torch import Tensor
import torchaudio
from torchaudio.compliance.kaldi import fbank
from typing import Union
from argparse import Namespace

from .beam_search import beam_search
from .model import get_model, CifRnntModel
from .tokenizer import Tokenizer
from .utils import best_path_search

FEATURE_DIM = 80

CiftModel_config = {
  "num_encoder_layers": "2,2,3,4,3,2",
  "downsampling_factor": "1,2,4,8,4,2",
  "feedforward_dim": "512,768,1024,1536,1024,768",
  "num_heads": "4,4,4,8,4,4",
  "encoder_dim": "192,256,384,512,384,256",
  "query_head_dim": "32",
  "value_head_dim": "12",
  "pos_head_dim": "4",
  "pos_dim": 48,
  "encoder_unmasked_dim": "192,192,256,256,256,192",
  "cnn_module_kernel": "31,31,15,15,15,31",
  "causal": True,
  "chunk_size": "-1",
  "left_context_frames": "-1",
  "output_downsampling_factor": 1,
  "feature_dim": FEATURE_DIM,
  "decoder_dim": 512,
  "context_size": 4,
  "joiner_dim": 512,
  "phi_type": "att;8",
  "phi_arch": "vanilla",
  "phi_norm_type": "layernorm",
  "alpha_actv": "abs",
  "omega_type": "Mean",
  "omega_init_scale": "D ** -0.5",
  "detach_alphas": 1,
  "use_ctc": 0,
  "prune_range": 16,
#   "blank_id": 0,
#   "unk_id": 2,
#   "vocab_size": 500,
}

torchaudio_compliance_kaldi_fbank_opts = {
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L43
    # "blackman_coeff": 0.42, # default value
    "channel": -1,
    # Icefall value
    # "dither": 0, default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L24
    "energy_floor": 0.,
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L33
    # "frame_length": 25., # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L32
    # "frame_shift": 10., # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/mel-computations.h#L23
    # "high_freq": 0., # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L32
    # "htk_compat": False, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/mel-computations.h#L19
    # "low_freq": 20., # default value
    # Not in kaldifeat
    # "min_duration": 0.,
    # Icefall value
    "num_mel_bins": FEATURE_DIM,
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L35
    # "preemphasis_coefficient": 0.97, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L28
    # "raw_energy": True, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L36
    # "remove_dc_offset": True, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L42
    # "round_to_power_of_two": True, # default value
    # Icefall value
    # "sample_frequency": 16000.0, # default value
    # Icefall value
    "snip_edges": False,
    # Not in kaldifeat
    # "subtract_mean": False, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L23
    # "use_energy": False, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L35
    # "use_log_fbank": True, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.h#L39
    # "use_power": True, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/mel-computations.h#L29
    # "vtln_high": -500.0, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/mel-computations.h#L25
    # "vtln_low": 100.0, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-fbank.cc#L23
    # "vtln_warp": 1.0, # default value
    # https://github.com/csukuangfj/kaldifeat/blob/c1aefb643ee6352b8bd3123c2f5b021f1a1aa57f/kaldifeat/csrc/feature-window.h#L37
    # "window_type": "povey", # default value
}
    
_PAD_VALUE = -23.025850929940457 # math.log(1e-10)
_SAMPLE_RATE = 16000

class CifRnntWrapper:
    def __init__(
        self,
        model_filepath : str,
        lang_dir : str,
        device : Union[str, torch.device],
    ):
        torch.set_grad_enabled(False)

        self.tokenizer = Tokenizer.load(lang_dir=lang_dir)
        
        model_configs = {
            "blank_id": self.tokenizer.piece_to_id("<blk>"),
            "unk_id": self.tokenizer.piece_to_id("<unk>"),
            "vocab_size": self.tokenizer.get_piece_size(),
            **CiftModel_config,
        }

        model_configs = Namespace(**model_configs)
        model : CifRnntModel = get_model(model_configs)
        model = model.to(device)

        model_params = torch.load(model_filepath, map_location=device)
        model.load_state_dict(model_params["model"])
        self.model = model.eval()     

        self.device = device

    def _check_mel(self, mel : Tensor, mel_lens : Tensor) -> tuple[Tensor, Tensor]:
        
        assert mel.ndim == 3, (
            f"Invalid mel.ndim {mel.ndim}. Expected 3."
        )
        
        B, T, F = mel.shape
        assert F == FEATURE_DIM
        assert B == mel_lens.size(0), (
                f"Expected {B} elements in mel_lens. Got {mel_lens.size(0)}"
            )
        
        return mel, mel_lens

    def mel_to_awe(
        self,
        mel : Tensor,
        mel_lens : Tensor
    ) -> tuple[Tensor, Tensor]:
        mel, mel_lens = self._check_mel(mel, mel_lens)

        encoder_out, encoder_out_lens, feature_mask = self.model.forward_encoder(
            mel, mel_lens
        )
        alphas = self.model.omega(encoder_out, encoder_out_lens)
        awe, awe_lens = self.model.phi(encoder_out, encoder_out_lens, alphas, feature_mask)
        
        return awe, awe_lens

    def awe_to_text(
        self,
        awe : Tensor,
        awe_lens : Tensor,
        beam_size : int = 4,
        max_s_per_t : int = 9,
    ):
        hyps = beam_search(
            model=self.model,
            encoder_out=awe,
            encoder_out_lens=awe_lens,
            beam=beam_size,
            max_s_per_t=max_s_per_t,
        )
        hyps = self.tokenizer.decode(hyps)
        return hyps

    def mel_to_text(
        self,
        mel : Tensor,
        mel_lens : Tensor,
        beam_size : int = 4,
        max_s_per_t : int = 9,
    ):
        awe, awe_lens = self.mel_to_awe(mel, mel_lens)
        hyps = self.awe_to_text(awe, awe_lens, beam_size, max_s_per_t)
        return hyps

    def awe_text_to_alignment(
        self,
        awe : Tensor,
        awe_lens : Tensor,
        text : list[str],
    ):       
        tokens = self.tokenizer.encode(text)
        
        B = awe.size(0)
        
        ret = []

        for i in range(B):
            hyp_w_blank = best_path_search(
                model=self.model,
                encoder_out=awe[i, None, :awe_lens[i]],
                tokens=tokens[i],
            )
            hyp_w_blank = self.tokenizer.decode(hyp_w_blank)
            hyp_w_blank = hyp_w_blank.split("<blk>")[:-1]
            ret.append(hyp_w_blank)
        
        return ret

    def wav_to_mel(self, waves : list[Tensor]) -> tuple[Tensor, Tensor]:
        waves =  [w.to(self.device) for w in waves]
    
        mel = [fbank(waveform=w, **torchaudio_compliance_kaldi_fbank_opts) for w in waves]
        mel_lens = [m.size(0) for m in mel]

        mel = torch.nn.utils.rnn.pad_sequence(mel, batch_first=True, padding_value=_PAD_VALUE)
        mel_lens = torch.tensor(mel_lens, device=self.device)

        return mel, mel_lens

    def file_to_wav(self, *filenames : list[str]) -> list[Tensor]:
        ret : list[Tensor] = []
        
        for f in filenames:
            wave, sample_rate = torchaudio.load(f)
            assert (
                sample_rate == _SAMPLE_RATE
            ), f"Expected sample rate: {_SAMPLE_RATE}. {f} given: {sample_rate}"
            assert (
                wave.size(0) == 1
            ), f"Expected mono-channel. Given {wave.size(0)} channels."
            ret.append(wave[:1])
        
        return ret
