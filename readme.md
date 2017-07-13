# train and save a model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=train # note that if using lenet model, you should use mnist_conv (which has correct input size)

# calculate and save statistics for that model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=compute_stats --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000

# optimize a dataset using the saved model and the statistics
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=optimize_dataset --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000 --optimization_objective=top_layer # or all_layers, spectral_all_layers, spectral_layer_pairs

# distilling a dataset using vanilla knowledge distillation
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=distill --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000 --eval_dataset=mnist --student_model=hinton1200

# distilling a dataset using one of the reconstructed datasets:
python main.py --run_name=test_train_works --dataset=summaries/test_train_works/data/data_optimized_all_layers_test_train_works.npy --model=hinton1200 --procedure=distill --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000 --eval_dataset=mnist --student_model=hinton1200 --epochs=30



TODO(rapha):
- implement fixed dropout filters again...
- random scrips and viz scripts

TODO(sfenu3): spectral optimization objectives
(search for "TODO(sfenu3"))


for conv model:

python main.py --run_name=test_train_works --dataset=mnist_conv --model=lenet --procedure=train

python main.py --run_name=test_train_works --dataset=mnist_conv --model=lenet --procedure=compute_stats --model_meta=summaries/test_train_works/train/checkpoint/lenet-8000.meta --model_checkpoint=summaries/test_train_works/train/checkpoint/lenet-8000

python main.py --run_name=test_train_works --dataset=mnist_conv --model=lenet --procedure=optimize_dataset --model_meta=summaries/test_train_works/train/checkpoint/lenet-8000.meta --model_checkpoint=summaries/test_train_works/train/checkpoint/lenet-8000 --optimization_objective=top_layer # or all_layers, spectral_all_layers, spectral_layer_pairs

python main.py --run_name=test_train_works --dataset=summaries/test_train_works/data/data_optimized_top_layer_test_train_works.npy --model=lenet --procedure=distill --model_meta=summaries/test_train_works/train/checkpoint/lenet-8000.meta --model_checkpoint=summaries/test_train_works/train/checkpoint/lenet-8000 --eval_dataset=mnist_conv --student_model=lenet_half --epochs=30
