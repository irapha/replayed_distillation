#!/bin/bash
# scp -r rapha@208.104.138.86:~/rapha/replayed_distillation/summaries/$1/* summaries/$1/
mkdir summaries/$1
mkdir summaries/$1/train
mkdir summaries/$1/train_batch
mkdir summaries/$1/test
scp -r beetle.raphagl.com:~/dev/replayed_distillation/summaries/$1/train/* summaries/$1/train
scp -r beetle.raphagl.com:~/dev/replayed_distillation/summaries/$1/train_batch/* summaries/$1/train_batch
scp -r beetle.raphagl.com:~/dev/replayed_distillation/summaries/$1/test/* summaries/$1/test
