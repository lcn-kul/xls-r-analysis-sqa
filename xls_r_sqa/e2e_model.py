from pathlib import Path
import torch
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Iterable, Optional, Tuple, Union
import warnings

from xls_r_sqa.center_crop import CenterCrop
from xls_r_sqa.config import Config, FEAT_SEQ_LEN
from xls_r_sqa.sqa_model import FusionModel, SingleLayerModel
from xls_r_sqa.utils import get_model_path, download_truncated_xlsr

class E2EModel(torch.nn.Module):
    def __init__(
            self,
            config: Config,
            xlsr_layers: Optional[Union[int, Iterable[int]]] = None,
            xlsr_dir: Optional[Path] = None,
            sqa_path: Optional[Path] = None,
            dataset_variant: str = "subset",
            auto_download: bool = True,
    ):
        super().__init__()

        # Parse config.
        self.config = config
        self.use_xlsr = "xlsr" in config.name.lower()
        self.sample_rate = 16000

        # If user didn't specify paths, generate default paths.
        if xlsr_dir is None or sqa_path is None:
            xlsr_dir, sqa_path = get_model_path(config, xlsr_layers, dataset_variant)

        # Parse xlsr_layers.
        if self.use_xlsr:
            if xlsr_layers is None:
                raise ValueError("Please specify xlsr_layers!")
            if isinstance(xlsr_layers, int):
                xlsr_layers = (xlsr_layers,)
            if not (isinstance(xlsr_layers, Iterable) and len(xlsr_layers) in [1,2]):
                raise Exception("xlsr_layers must either be one or two integers")
            self.xlsr_layers = xlsr_layers
            self.is_fusion = (len(xlsr_layers) == 2)

            # If not present, optionally attempt a download
            if not xlsr_dir.is_dir():
                if auto_download:
                    # Reconstruct model_size, e.g. "300m-lay5"
                    name_lower = config.name.lower()
                    if "300m" in name_lower:
                        size_str = "300m"
                    elif "1b" in name_lower:
                        size_str = "1b"
                    elif "2b" in name_lower:
                        size_str = "2b"
                    else:
                        raise ValueError("Unrecognized XLS-R config name.")
                    max_layer = max(xlsr_layers)
                    model_size = f"{size_str}-lay{max_layer}"

                    print(f"[INFO] Attempting to download truncated XLS-R: {model_size}")
                    download_truncated_xlsr(model_size)

                    # If still missing, we'll raise an error below
                    if not xlsr_dir.is_dir():
                        raise FileNotFoundError(
                            f"Auto-download failed or incomplete. "
                            f"Could not find folder {xlsr_dir}."
                        )
                else:
                    warnings.warn(
                        f"Truncated XLS-R dir not found at: {xlsr_dir}\n"
                        "Run `download_truncated_xlsr(...)` or pass `auto_download=True` "
                        "when creating E2EModel to fetch automatically."
                    )

            # XLS-R feature extractor.
            self.w2v2_feat_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=self.sample_rate,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
            self.xlsr_model = Wav2Vec2Model.from_pretrained(str(xlsr_dir))
        else: 
            # MFCC-based.
            self.xlsr_layers = None
            self.is_fusion = False
            self.mfcc_calculator = MFCC(sample_rate=self.sample_rate)

        # Center crop.
        self.crop = CenterCrop(FEAT_SEQ_LEN)

        # SQA model.
        if self.is_fusion:
            self.sqa_model = FusionModel(config)
        else:
            self.sqa_model = SingleLayerModel(config)
        self.sqa_model.load_state_dict(torch.load(sqa_path, map_location="cpu"))


    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        r"""Predict perceived speech quality in [1,5] for the given audio sample.

        Args:
            audio: audio sample to process.

        Shape:
            - features: (N,T) or (T,)
            - output: (N,1) or (1,)
        """

        unsqueezed = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # Include batch dim
            unsqueezed = True

        if audio.size(1) > 15 * self.sample_rate:
            warnings.warn("Warning: audio is longer than 15 seconds. It might be better to compute predictions over a sliding window and take the average.")

        # Calculate XLS-R / MFCC features.
        if self.use_xlsr:
            audio_np = audio.detach().cpu().numpy()
            xlsr_input = self.w2v2_feat_extractor(
                audio_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )["input_values"]
            if xlsr_input.dim() == 3:
                xlsr_input = xlsr_input.squeeze(0) # compatability old/new huggingface transformers
            xlsr_input = xlsr_input.to(audio.device)
            with torch.no_grad():
                xlsr_out = self.xlsr_model(xlsr_input, output_hidden_states=True)
                x = [self.crop(xlsr_out.hidden_states[i].squeeze(0)) for i in self.xlsr_layers]
                if len(self.xlsr_layers) == 1:
                    x = x[0] # Extract from list for SingleLayerModel
        else:
            mfcc: torch.Tensor = self.mfcc_calculator(audio)
            mfcc = mfcc.transpose(-1, -2) # transpose from (N, n_mfcc, L) to (N, L, n_mfcc)
            with torch.no_grad():
                x = self.crop(mfcc.to(audio.device))

        # Calculate MOS prediction.
        sqa_out = self.sqa_model(x)
        sqa_out_denorm = 1.0 + sqa_out * 4.0
        if unsqueezed:
            sqa_out_denorm = sqa_out_denorm.squeeze(0)
        return sqa_out_denorm

