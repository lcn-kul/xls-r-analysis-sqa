import os
import torch

### This file was used to truncate the pre-trained XLS-R files for faster inference.

def truncate_w2v2(pretrained_bin: str, layers_to_keep: int, num_layers):
    """
    Note: we will keep one extra layer because there's an extra layer_norm after the
    final layer. Make sure you use the second-to-last hidden state!
    See: https://github.com/huggingface/transformers/blob/31d452c68b34c2567b62924ee0df40a83cbc52d5/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L894
    
    Args:
        - pretrained_bin : str Path to "pytorch_model.bin" file of local clone of
          Wav2Vec2 model.
        - layers_to_keep: int Number of transformer layers to keep.
        - num_layers: int Total number of transformer layers in pre-trained model (24
          or 48).

    """

    _model_dict = torch.load(pretrained_bin)
    for idx in range(layers_to_keep+1, num_layers):
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.k_proj.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.k_proj.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.v_proj.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.v_proj.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.q_proj.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.q_proj.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.out_proj.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.attention.out_proj.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.layer_norm.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.layer_norm.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.feed_forward.intermediate_dense.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.feed_forward.intermediate_dense.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.feed_forward.output_dense.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.feed_forward.output_dense.bias"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.final_layer_norm.weight"]
        del _model_dict[f"wav2vec2.encoder.layers.{idx}.final_layer_norm.bias"]

    # Remove these as well to avoid warning when loading.
    del _model_dict["project_hid.weight"]
    del _model_dict["project_hid.bias"]
    del _model_dict["project_q.weight"]
    del _model_dict["project_q.bias"]
    del _model_dict["quantizer.codevectors"]
    del _model_dict["quantizer.weight_proj.weight"]
    del _model_dict["quantizer.weight_proj.bias"]

    return _model_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    pretrained_basedir = os.path.join(script_dir, "models", "xls-r")
    truncated_basedir = os.path.join(script_dir, "models", "xls-r-trunc")
    xlsr_names = ["wav2vec2-xls-r-300m", "wav2vec2-xls-r-1b", "wav2vec2-xls-r-2b"]
    xlsr_bins = {x: os.path.join(pretrained_basedir, x, "pytorch_model.bin") for x in xlsr_names}
    to_truncate = [
        ("wav2vec2-xls-r-300m", 5, 24), ("wav2vec2-xls-r-300m", 21, 24),
        ("wav2vec2-xls-r-1b", 10, 48), ("wav2vec2-xls-r-1b", 41, 48),
        ("wav2vec2-xls-r-2b", 10, 48), ("wav2vec2-xls-r-2b", 41, 48),
    ]
    for xlsr_name, layers_to_keep, num_layers in to_truncate:
        print(f"Truncating {xlsr_name} to {layers_to_keep} transformer layers...")
        model_dict = truncate_w2v2(xlsr_bins[xlsr_name], layers_to_keep, num_layers)
        new_dir_name = xlsr_name + "-lay" + str(layers_to_keep)
        out_path = os.path.join(truncated_basedir, new_dir_name, "pytorch_model.bin")
        torch.save(model_dict, out_path)
    print("Finished.")

