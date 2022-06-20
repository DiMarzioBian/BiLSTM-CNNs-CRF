# BiLSTM-CNNs-CRF

## Warning: this repo implements batch processs to accelerate, but I am new to NLP, the character padding and word padding are not written in good gramma, which leads to performance downgrade, F1: 0.81. The original repo use 1 batch size and acheived F1: 0.92.

This is a PyTorch implementation for the ACL'16 paper [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://aclanthology.org/P16-1101/)



####  Download GloVe vectors and extract glove.6B.100d.txt into "./data/raw/" folder

`wget http://nlp.stanford.edu/data/glove.6B.zip`

#### Download corpus file into "./data/raw/" folder

You can download the data files from within this repo [**over here**](https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data)

## Prepare data

Run `python prepare_data.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --seed                    random seed
  --word_dim                token embedding dimension
  --path_data               location of the data corpus
  --path_embedding          path to save embedding file
  --path_processed          path to save the processed data
  --path_filtered           path to save the filtered processed data
  
  --filter_word             filter meaningless words
  --tag_scheme              BIO or BIOES
  --is_lowercase            control lowercasing of words
  --digi_zero               control replacement of  all digits by
```

## Training model

Run `python main.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --seed                    random seed
  --device                  device for computing
  --path_data               path of the data corpus
  --path_processed          path to save the filtered processed data
  --path_filtered           optimizer type: Adam, AdamW, RMSprop, Adagrad, SGD
  --path_pretrained         path of the data corpus
  --path_model              path of the trained model
  --num_worker              number of dataloader worker
  --batch_size              batch size
  --epochs                  upper epoch limit
  --es_patience_max         max early stopped patience

  --dim_emb_char            character embedding dimension
  --dim_emb_word            word embedding dimension
  --dim_out_char            character encoder output dimension
  --dim_out_word            word encoder output dimension
  --window_kernel           window width of CNN kernel

  --enable_pretrained       use pretrained glove dimension
  --freeze_glove            free pretrained glove embedding
  --dropout                 dropout rate applied to layers (0 = no dropout)
  --lr                      initial learning rate
  --lr_step                 number of epoch for each lr downgrade
  --lr_gamma                strength of lr downgrade
  --eps_f1                  minimum f1 score difference threshold

  --mode_char               character encoder: lstm or cnn
  --mode_word               word encoder: lstm or cnn1, cnn2, cnn3, cnn_d
  --enable_crf              employ CRF
  --filter_word             filter meaningless words
```
No arguments will run the model in the settings that achieved best result.

## File structure
```bash
----BiLSTM-CNNs-CRF\
    |----data\
    |    |----raw\
    |    |    |----eng.testa
    |    |    |----eng.testb
    |    |    |----eng.train
    |    |    |----glove.6B.100d.txt
    |    |----data_bundle.pkl
    |    |----data_filtered_bundle.pkl
    |----data\
    |    |----models\
    |    |    |----model.pt
    |----dataloader.py
    |----epoch.py
    |----main.py
    |----model.py
    |----prepare_data.py
    |----README.md
    |----utils.py
```
