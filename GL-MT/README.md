# GL-MT system

See in config.py the different paths if you want to modify them for your own data.

## Train SED model

- `python main.py`

## Test models

```bash
python TestModel.py -m "model_path" -g ../dataset/metadata/validation/validation.tsv  \
-ga ../dataset/audio/validation -s stored_data/gl-mt/validation_predictions.tsv 
```
