import os
import librosa
import numpy as np
import soundfile as sf
import torch

from xls_r_sqa.config import (
    MFCC_TRANSFORMER_32DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG,
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
)
from xls_r_sqa.e2e_model import E2EModel
from xls_r_sqa.utils import get_model_path

def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=new_sr,
            res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


def load_audio(file_path: str, sampling_rate: int) -> torch.Tensor:
    array, _ = _decode_non_mp3_file_like(file_path, sampling_rate)
    array = np.float32(array)
    return array



def test_e2e():

    # ======== **SELECT CONFIG** ======== #
    # - MFCC_TRANSFORMER_32DEEP_CONFIG
    # - XLSR_300M_TRANSFORMER_32DEEP_CONFIG
    # - XLSR_1B_TRANSFORMER_32DEEP_CONFIG
    # - XLSR_2B_TRANSFORMER_32DEEP_CONFIG
    config = XLSR_2B_TRANSFORMER_32DEEP_CONFIG
    use_xlsr = "XLSR" in config.name

    # ======== **SELECT LAYERS** ======== #
    # - XLSR_300M:  5, 21, or [ 5,21]
    # - XLSR_1B  : 10, 41, or [10,41]
    # - XLSR_2B  : 10, 41, or [10,41]
    if use_xlsr:
        xlsr_layers = 10
    else:
        xlsr_layers = None  # For MFCC-based models

    # ======== **SELECT DATASET** ======= #
    # - "full" or "subset"
    ds = "subset"

    # ============ EXECUTION ============ #

    # Create end-to-end model.
    print("Loading model...")
    device = "cpu"
    e2e_model = E2EModel(config, xlsr_layers, dataset_variant=ds, auto_download=True)
    e2e_model = e2e_model.to(device)
    e2e_model.eval()

    # Run inference on files.
    print("Running inference on files...")
    script_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(script_dir, "audio_samples")
    file_names = [f"iub-{x}.wav" for x in ["bad", "poor", "fair", "good", "excellent"]]
    for file_name in file_names:
        print(f"Prediction: {file_name}")
        file_path = os.path.join(audio_dir, file_name)
        audio_np = load_audio(file_path, sampling_rate=16000)
        audio_pt = torch.from_numpy(audio_np).to(device)
        mos_pred = e2e_model.forward(audio_pt)
        print("%0.6f" % mos_pred.item())

    print("Finished.")


if __name__ == "__main__":
    test_e2e()
