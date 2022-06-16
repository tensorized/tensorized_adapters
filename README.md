# Tensorized Adapters: A Parameter Efficient Regularization Approach 

TL;DR: Tensorized Layers are a drop-in replacement of adapters for fine-tuning pre-trained language models and they naturally exploit the low rank structure of the adapter layers.

Keywords: tensorized layers, adapters, parameter-efficiency

Abstract: Adapters, Compactors, and BitFit are parameter-efficient approaches to fine-tuning large language models by freezing the entire language model and adding task-specific, fully-connected layers. The goals are two-fold: Parameter efficiency and improved task regularization. Tensorized layers (CP, Tucker, TensorTrain, and TensorRing) are more efficient and have also been shown to act as regularizers. One natural extension is using Tensorized layers to parameterize the adapter layers and simultaneously exploit parameter efficiency and regularization ability. This work extensively studies  Tensorized layers that can naturally exploit the low-rank structure of the adapter layers and provide a drop-in replacement for Adapters. We empirically investigate the different tensor decomposition approaches and the impact of tensor ranks in the adapter layers. This approach reduces the number of parameters of adapters by up to 5 times with no drop in performance. 

# Environment

Transformers version: "4.20.0.dev0"

Tensorly version: "0.7.0"


# Running Instruction

```
python run_glue_tensorized.py   --model_name_or_path bert-base-cased   --task_name mrpc   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-4   --num_train_epochs 20   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir --adapter_projection_type cp --tensorized_layer_rank 4 --if_adapter_like True --adapter_size 64
```


|                                  |   MRPC |     RTE |    STSB |
|:---------------------------------|-------:|--------:|--------:|
| ('baselines', 'Adapters', '-')   |  0.896 |   0.688 |   0.873 |
| ('baselines', 'BitFit', '-')     |  0.904 |   0.723 |   0.892 |
| ('baselines', 'Diff-Prune', '-') |  0.897 |   0.706 |   0.86  |
| ('baselines', 'Full-FT', '-')    |  0.89  |   0.705 |   0.889 |
| ('tensorized', 'cp', 2.0)        |  0.906 |   0.682 |   0.883 |
| ('tensorized', 'cp', 4.0)        |  0.906 |   0.704 |   0.886 |
| ('tensorized', 'tt', 2.0)        |  0.905 |   0.682 |   0.885 |
| ('tensorized', 'tt', 4.0)        |  0.909 |   0.711 |   0.886 |
| ('tensorized', 'tucker', 2.0)    |  0.903 |   0.701 |   0.872 |
| ('tensorized', 'tucker', 4.0)    |  0.904 |   0.711 |   0.881 |
