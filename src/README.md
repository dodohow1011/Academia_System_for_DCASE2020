# Baseline

See in config.py the different paths if you want to modify them for your own data.

## Train SED model without source separation

- `python main.py`

## Testing baseline models
### SED only
```bash
python TestModel.py -m "model_path" -g ../dataset/metadata/validation/validation.tsv  \
-ga ../dataset/audio/validation -s stored_data/baseline/validation_predictions.tsv 
```
