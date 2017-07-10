# train and save a model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=train

# calculate and save statistics for that model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=compute_stats --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000

# optimize a dataset using the saved model and the statistics
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=optimize_dataset --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000 --optimization_objective=top_layer # or all_layers, spectral_all_layers, spectral_layer_pairs
