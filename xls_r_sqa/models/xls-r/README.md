# xls-r

Truncated XLS-R models are used in this repository. These can be downloaded directly
[[link]](/xls_r_sqa/models/xls-r-trunc/README.md), or by downloading the full pre-trained XLS-R
models and running the script [`truncate_w2v2.py`](/truncate_w2v2.py).


 **Download Pre-Trained XLS-R Models**

```
git lfs install
git clone --depth 1 https://huggingface.co/facebook/wav2vec2-xls-r-300m
git clone --depth 1 https://huggingface.co/facebook/wav2vec2-xls-r-1b
git clone --depth 1 https://huggingface.co/facebook/wav2vec2-xls-r-2b
```

This will take a large amount of space.
- wav2vec2-xls-r-300m = 2.4 GB
- wav2vec2-xls-r-1b = 7.2 GB
- wav2vec2-xls-r-2b = 18 GB
