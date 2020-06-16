# GL-MT system

See in config.py the different paths if you want to modify them for your own data.

# Pre-trained feature extractor

The pre-trained feature extractor is in `./stored_data/pretrained_model`

## Train SED model

- `python main.py`

The best model will be stored in `./stored_data/`

## Test models

```bash
python TestModel.py -m "model_path" -g "groudtruth_tsv" -ga "audio_path" -s "output_file_path"
```
e.g.
- `python TestModel.py -m stored_data/gl-mt/model/ms_best -g ../dataset/metadata/validation/validation.tsv \
-ga ../dataset/audio/validation -s stored_data/gl-mt/validation_predictions.tsv`

- `python TestModel.py -m stored_data/gl-mt/model/ms_best -g ../public_eval/metadata/eval/public.tsv \
-ga ../public_eval/audio/eval/public -s stored_data/gl-mt/eval_predictions.tsv`

## Result

**MS**

|         | Event-based    | Segment-based    |
----------|---------------:|-----------------:|
Validation| **45.68%**     | **71.96%**       |

**EMA**

|         | Event-based    | Segment-based    |
----------|---------------:|-----------------:|
Validation| **45.65%**     | **71.87%**       |
