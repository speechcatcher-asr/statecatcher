# usage: train_sentencepiece.py [-h] --input INPUT [--model_prefix MODEL_PREFIX] [--vocab_size VOCAB_SIZE] [--character_coverage CHARACTER_COVERAGE] [--model_type {unigram,bpe,char,word}] [--unk_piece UNK_PIECE] [--pad_piece PAD_PIECE] [--bos_piece BOS_PIECE] [--eos_piece EOS_PIECE]
# train_sentencepiece.py: error: the following arguments are required: --input

python3 train_sentencepiece.py --input data_en_33k_lm_train.txt --vocab_size 1024 --model_type bpe --character_coverage 0.995
