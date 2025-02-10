import os
from pathlib import Path
import subprocess
from typing import Iterable, Optional, Tuple, Union
import warnings

from xls_r_sqa.config import Config

# Assume models are stored in xls_r_sqa/models/
MODEL_DIR = Path(__file__).parent / "models"

def get_model_path(
    config: Config,
    xlsr_layers: Optional[Union[int, Iterable[int]]] = None,
    dataset_variant: str = "subset"
) -> Tuple[Optional[Path], Path]:
    """
    Returns (xlsr_dir, sqa_path) for the given config & layers.
    The xlsr_dir is None if using MFCC-based model, otherwise it points
    to the truncated XLS-R directory. The sqa_path points to the SQA
    model weights for either MFCC or XLS-R.

    Parameters
    ----------
    config : Config
        Configuration object containing model name and other hyperparameters.
    xlsr_layers : int or Iterable[int], optional
        One or two XLS-R layer indices if using an XLS-R-based config. 
        Ignored for MFCC-based configs.
    dataset_variant : str, optional
        Dataset subset or variant (e.g., "subset", "full", etc.). 
        Used to pick the correct model checkpoint name.

    Returns
    -------
    (Path or None, Path)
        A tuple (xlsr_dir, sqa_path):
          - xlsr_dir: None if MFCC-based, else a Path to the truncated XLS-R folder.
          - sqa_path: Path to the SQA model .pt weights file.

    Raises
    ------
    ValueError
        If the config name is unrecognized for XLS-R,
        or if xlsr_layers is missing or invalid for XLS-R configs.
    """
    name_lower = config.name.lower()
    use_xlsr = "xlsr" in name_lower

    if use_xlsr:
        # Extract e.g. "1b" or "2b" from config.name
        # Customize this logic to match how you name your directories
        # Example: "XLSR_300M_TRANSFORMER_32DEEP" -> "300m"
        if "300m" in name_lower:
            xlsr_size = "300m"
            possible_layers = [5, 21]
        elif "1b" in name_lower:
            xlsr_size = "1b"
            possible_layers = [10, 41]
        elif "2b" in name_lower:
            xlsr_size = "2b"
            possible_layers = [10, 41]
        else:
            raise ValueError("Unrecognized XLS-R config name.")

        if xlsr_layers is None:
            raise ValueError("Please provide xlsr_layers for XLS-R configs.")
        if isinstance(xlsr_layers, int):
            xlsr_layers = [xlsr_layers]
        for layer in xlsr_layers:
            if layer not in possible_layers:
                raise ValueError(
                    f"Invalid layer {layer}! Possible layers for XLS-R {xlsr_size}: {possible_layers}"
                )

        # Truncation directory name
        max_layer = max(xlsr_layers)
        xlsr_dir = MODEL_DIR / "xls-r-trunc" / f"wav2vec2-xls-r-{xlsr_size}-lay{max_layer}"

        # For SQA model name
        is_fusion = (len(xlsr_layers) == 2)
        if is_fusion:
            fusion_name = "fusion"
        else:
            fusion_name = f"lay{xlsr_layers[0]}"

        sqa_path = MODEL_DIR / "sqa-v2" / f"xls-r-{xlsr_size}" / f"model_{xlsr_size}_{fusion_name}_{dataset_variant}.pt"
        return xlsr_dir, sqa_path
    else:
        # MFCC-based
        xlsr_dir = None
        sqa_path = MODEL_DIR / "sqa-v2" / "mfcc" / f"model_mfcc_{dataset_variant}.pt"
        return xlsr_dir, sqa_path

###############################################################################
# Downloading Truncated XLS-R from Hugging Face
###############################################################################

# A small lookup table from your instructions:
# '300m-lay5'   => 0.7 GB
# '300m-lay21'  => 2.2 GB
# '1b-lay10'    => 1.8 GB
# '1b-lay41'    => 6.4 GB
# '2b-lay10'    => 4.0 GB
# '2b-lay41'    => 14.2 GB
TRUNCATED_MODEL_REPOS = {
    "300m-lay5":   "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-300m-lay5",
    "300m-lay21":  "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-300m-lay21",
    "1b-lay10":    "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-1b-lay10",
    "1b-lay41":    "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-1b-lay41",
    "2b-lay10":    "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-2b-lay10",
    "2b-lay41":    "https://huggingface.co/kul-speech-lab/wav2vec2-xls-r-2b-lay41",
}

def download_truncated_xlsr(
    model_size: str,
    target_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Download (via Git + Git LFS) the truncated XLS-R model(s) from Hugging Face
    into the target directory.

    Parameters
    ----------
    model_size : str
        Which truncated XLS-R model to download. Must be one of:
            '300m-lay5', '300m-lay21',
            '1b-lay10',  '1b-lay41',
            '2b-lay10',  '2b-lay41'.
    target_dir : str, Path, or None
        Directory where the repository is cloned. If None, the default location
        is "xls_r_sqa/models/xls-r-trunc/" relative to this file. For example:
        'xls_r_sqa/models/xls-r-trunc/'. If the directory doesn't exist,
        it will be created.

    Example
    -------
    >>> from xls_r_sqa.utils import download_truncated_xlsr
    >>> download_truncated_xlsr('300m-lay5')  # uses default directory
    >>> download_truncated_xlsr('300m-lay5', 'my_custom_folder')  # custom path

    Notes
    -----
    - You must have 'git' and 'git lfs' installed for this to work.
    - These repos can be large (0.7 GB up to 14.2 GB).
    - The user must ensure there's enough free disk space.
    """

    if model_size not in TRUNCATED_MODEL_REPOS:
        valid_keys = ", ".join(sorted(TRUNCATED_MODEL_REPOS.keys()))
        raise ValueError(
            f"Unsupported model_size: {model_size}. "
            f"Available options are: {valid_keys}"
        )

    # If no target_dir is provided, default to xls_r_sqa/models/xls-r-trunc/
    if target_dir is None:
        target_dir = Path(__file__).parent / "models" / "xls-r-trunc"

    repo_url = TRUNCATED_MODEL_REPOS[model_size]
    target_dir = Path(target_dir).expanduser().resolve()
    clone_dir = target_dir / f"wav2vec2-xls-r-{model_size}"

    if clone_dir.exists():
        warnings.warn(
            f"Directory already exists: {clone_dir}\n"
            "Skipping clone. If you want a fresh clone, please remove "
            "the existing folder or choose a different target directory."
        )
        return

    # Print a caution about large sizes
    print(
        f"[INFO] Downloading truncated XLS-R model '{model_size}' from:\n"
        f"  {repo_url}\n"
        f"into:\n"
        f"  {clone_dir}\n\n"
        "Please ensure you have 'git' and 'git lfs' installed.\n"
        "This could be large (up to 14+ GB for '2b-lay41')."
    )

    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # 1) Attempt: git lfs install
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
    except FileNotFoundError:
        raise EnvironmentError(
            "Git or Git LFS not found. Please install git and git-lfs first.\n"
            "See https://git-lfs.github.com/ for instructions."
        )

    # 2) git clone --depth 1 ...
    cmd = ["git", "clone", "--depth", "1", repo_url, str(clone_dir)]
    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] Successfully cloned '{model_size}' into:\n  {clone_dir}\n")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Git clone failed. Command was:\n  {' '.join(cmd)}"
        ) from e
