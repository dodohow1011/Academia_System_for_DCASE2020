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
- `python TestModel.py -m stored_data/gl-mt/model/ms_best -g ../dataset/metadata/validation/validation.tsv -ga ../dataset/audio/validation -s stored_data/gl-mt/validation_predictions.tsv`

- `python TestModel.py -m stored_data/gl-mt/model/ms_best -g ../public_eval/metadata/eval/public.tsv -ga ../public_eval/audio/eval/public -s stored_data/gl-mt/eval_predictions.tsv`

## Result

System performance are reported in term of event-based F-scores with a 200ms collar on onsets and a 200ms / 20% of the events length collar on offsets.

**MS**

|         | Event-based    | Segment-based    |
----------|---------------:|-----------------:|
Validation| **45.68%**     | **71.96%**       |
----------|---------------:|-----------------:|
Evaluation| **47.47%**     | **74.63%**       |

**EMA**

|         | Event-based    | Segment-based    |
----------|---------------:|-----------------:|
Validation| **45.65%**     | **71.87%**       |
----------|---------------:|-----------------:|
Evaluation| **48.50%**     | **75.83%**       |
