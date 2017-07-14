# Data-Free Knowledge Distillation For Deep Neural Networks

On arxiv soon!

# Requirements

This code requires that you have [tensorflow](https://tensorflow.org) 1.0 installed, along with `numpy`
and `scikit-image 0.13.0` on python 3.6+.

The visualization scripts (used to debug optimized/reconstructed datasets) also
require `opencv 3.2.0` and `matplotlib`.

# Usage

We provide two teacher model implementations and two student model implementations.


### Train and Save a Model
The `procedure` flag specifies what to do with the model and dataset. In this
case, train it from scratch.

```bash
python main.py --run_name=experiment --model=hinton1200 --dataset=mnist \
    --procedure=train
```

### Calculate and Save Statistics for that Model
The `model_meta` and `model_checkpoint` flags are required because the
`compute_stats` procedure loads a pre-trained model.

```bash
python main.py --run_name=experiment --model=hinton1200 --dataset=mnist \
    --procedure=compute_stats \
    --model_meta=summaries/experiment/checkpoint/hinton1200-8000.meta \
    --model_checkpoint=summaries/experiment/checkpoint/hinton1200-8000
```

### Optimize a Dataset Using the Saved Model and the Statistics
This is where the real magic happens. The pre-trained model is loaded, and a
new graph is constructed using its saved weights, but as constants. This
ensures that the only thing being back-propagated to is the input
`tf.Variable`. The `optimization_objective` flag is needed to determine what
loss to use (see paper for details, coming soon on arxiv). The `dataset` flag
is only needed to determine `io_size`, so if you're using a pre-trained model
you don't have the original data for, you can mock the dataset class and simply
provide the `self.io_size` attribute. Using all of this, a new dataset will be
reconstructed and saved.

```bash
python main.py --run_name=experiment --model=hinton1200 --dataset=mnist \
    --procedure=optimize_dataset \
    --model_meta=summaries/experiment/checkpoint/hinton1200-8000.meta \
    --model_checkpoint=summaries/experiment/checkpoint/hinton1200-8000 \
    --optimization_objective=top_layer
    # or all_layers, spectral_all_layers, spectral_layer_pairs
```

### Distilling a Dataset Using One of the Reconstructed Datasets
You can then train a student network on the reconstructed dataset, and the
temperature-scaled teacher model activations. This time, the `dataset` flag is
the location where the reconstructed dataset was saved. Additionally, a
`student_model` needs to be specified to be trained from scratch. If you want
to evaluate the student's performance on the original test set (if you have
access to it), you can specify it as the `eval_dataset`.

```bash
python main.py --run_name=experiment --model=hinton1200 \
    --dataset=summaries/experiment/data/data_optimized_top_layer_experiment.npy \
    --procedure=distill \
    --model_meta=summaries/experiment/checkpoint/hinton1200-8000.meta \
    --model_checkpoint=summaries/experiment/checkpoint/hinton1200-8000 \
    --eval_dataset=mnist --student_model=hinton800 --epochs=30
```

### Distilling a Dataset Using Vanilla Knowledge Distillation
If you do have access to the original dataset, or you want to run Hinton's
original Knowledge Distillation [paper], you can just specify that `dataset`
flag.

```bash
python main.py --run_name=experiment --model=hinton1200 --dataset=mnist \
    --procedure=distill \
    --model_meta=summaries/experiment/checkpoint/hinton1200-8000.meta \
    --model_checkpoint=summaries/experiment/checkpoint/hinton1200-8000 \
    --eval_dataset=mnist --student_model=hinton800
```




TODO(rapha):
- implement fixed dropout filters again...
- random scrips and viz scripts
- readme and links and stuff

TODO(sfenu3): spectral optimization objectives
(search for "TODO(sfenu3"))


### Tips and Tricks

When using the lenet models, it should be noted that the [original paper]()
specified that mnist was resized from 28x28 pixel images to 32x32. Thus, then
using the convolutional models we provide, make sure to use the `mnist_conv`
dataset, which automatically resize the input images. The rest of the usage
should be exactly the same.

```bash
python main.py --run_name=experiment --model=lenet --dataset=mnist_conv \
    --procedure=train

python main.py --run_name=experiment --model=lenet --dataset=mnist_conv \
    --procedure=compute_stats \
    --model_meta=summaries/experiment/train/checkpoint/lenet-8000.meta \
    --model_checkpoint=summaries/experiment/train/checkpoint/lenet-8000

python main.py --run_name=experiment --model=lenet --dataset=mnist_conv \
    --procedure=optimize_dataset \
    --model_meta=summaries/experiment/train/checkpoint/lenet-8000.meta \
    --model_checkpoint=summaries/experiment/train/checkpoint/lenet-8000 \
    --optimization_objective=top_layer
    # or all_layers, spectral_all_layers, spectral_layer_pairs

python main.py --run_name=experiment --model=lenet \
    --dataset=summaries/experiment/data/data_optimized_top_layer_experiment.npy \
    --procedure=distill \
    --model_meta=summaries/experiment/train/checkpoint/lenet-8000.meta \
    --model_checkpoint=summaries/experiment/train/checkpoint/lenet-8000 \
    --eval_dataset=mnist_conv --student_model=lenet_half --epochs=30
```

# License

MIT
