#!/bin/bash

python3 rft_net.py \
  --train_event_file_path=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/sample-target.txt \
  --valid_event_file_path=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/sample-target.txt \
  --glove_file_path=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/glove.6B.300d.txt \
  --embedding_size=300 \
  --hidden_size=30 \
  --output_size=50 \
  --number_epochs=100 \
  --target_size=5 \
  --window=5 \
  --sample_size=5 \
  --learning_rate=1e-4 \
  --margin=0.5 \
  --print_every=100 \
  --rft_save_file=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/SavedModels/rft-test.ckpt \
  --embedder_save_file=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/SavedModels/embedder-test.ckpt \
  --indexer_and_bound_file=/home/jacobsuwang/Documents/UTA/Fall2018/CS395T/Project2/EventExtraction/SavedModels/indexer-and-bound-test.p