# GL-MT system

See in config.py the different paths if you want to modify them for your own data.

# Pre-trained feature extractor

The pre-trained feature extractor is in `./stored_data/pretrained_model`

## Train SED model

- `python main.py`

The best model will be stored in `./stored_data/`

## Test models

```bash
python TestModel.py -m "model_path" -g ../dataset/metadata/validation/validation.tsv  \
-ga ../dataset/audio/validation -s stored_data/gl-mt/validation_predictions.tsv 
```
