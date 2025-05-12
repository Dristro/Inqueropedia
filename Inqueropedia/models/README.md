# Models

Choose the model of your liking from this dir.

Refer to `notebooks/transformer_models.ipynb` for more info on the transformer models.
The notebook: `notebooks/preprocessing.ipynb` has the code for the bi-gram model.

NOTE: The bi-gram model will be used as a simple baseline for now.

The `DecoderOnlyTransformer.py` is the base file/script that will be used to build the custom-transformer models from here on.

All models and their specs:

|S.no|Name|Type|Desc|Tokenizer|train-loss|test-loss|
|-|-|-|-|-|-|-|
|1|Baseline|bi-gram|Simple baseline model to kick things off|bpe_tokenizer_v1_train_dataset|-|-|
