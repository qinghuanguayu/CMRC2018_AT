# CMRC2018_AT

## Usage

###  Step 1: Download BERT weights 
Download a BERT model and place it under the "model_bert" directory

```
model_bert
	--config.json
	--pytorch_model.bin
	--vocab.txt
```

- [bert-base-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin)
- [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base/resolve/main/pytorch_model.bin)
- [chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large/resolve/main/pytorch_model.bin)
- [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin)
- [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin)

### Step 2: Training and evaluating
```
python run_cmrc.py --model_type=bert --model_name_or_path=./model_bert --do_train --do_eval --do_lower_case --train_file=./data/cmrc2018_train.json --predict_file=./data/cmrc2018_dev.json --learning_rate=3e-5 --num_train_epochs=3 --max_seq_length=512 --doc_stride=128 --max_query_length=64 --per_gpu_train_batch_size=4 --per_gpu_eval_batch_size=6 --max_sentence_num=32  --warmup_steps=0.1 --ILF_rate=0.1 --output_dir=$Output_DIR --save_steps=1000 --eval_all_checkpoints --n_best_size=20 --max_answer_length=35 --gradient_accumulation_steps=1
```
