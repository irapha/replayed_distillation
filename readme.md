# train and save a model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=train

# calculate and save statistics for that model
python main.py --run_name=test_train_works --dataset=mnist --model=hinton1200 --procedure=compute_stats --model_meta=summaries/test_train_works/checkpoint/hinton1200-8000.meta --model_checkpoint=summaries/test_train_works/checkpoint/hinton1200-8000
