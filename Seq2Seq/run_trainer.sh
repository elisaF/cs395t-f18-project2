#!/bin/bash

python3 trainer.py \
  --data_dir=Data/ \
  --model_dir=SavedModels/ \
  --indexer_dir=SavedModels/ \
  --glove_path=Glove/glove.6B.300d.txt \
  --batch_size=32 \
  --number_batches=1000 \
  --number_epochs=10 \
  --embed_size=300 \
  --upsample_size=500 \
  --number_heads=5 \
  --number_blocks=4 \
  --factor=1 \
  --warmup=500 \
  --learning_rate=1e-4 \
  --dropout_rate=0.1 \
  --print_every=100 \
  --model_name=transformer \
  --validate=1