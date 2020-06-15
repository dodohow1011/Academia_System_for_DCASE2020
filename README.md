# DCASE2020 task4: Sound event detection in domestic environments using source separation

- Information about the DCASE 2020 challenge please visit the challenge [website].
- You can find discussion about the dcase challenge here: [dcase-discussions]. 

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit>=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, 
dcase_util >= 0.2.5, sed-eval >= 0.2.1, psds-eval >= 0.1.0, desed >= 1.1.7

## Environment

launch `conda_create_environment.sh` (recommended line by line)

### Reproducing the results
See [GL-MT] folder.

## Dataset

### Scripts to generate the dataset

In the [`scripts/`][scripts] folder, you can find the different steps to:
- Download recorded data and synthetic material.
- Generate synthetic soundscapes


**It is likely that you'll have download issues with the real recordings.
At the end of the download, please send a mail with the TSV files
created in the `missing_files` directory.** ([to Nicolas Turpault and Romain Serizel](#contact)).

However, if none of the audio files have been downloaded, it is probably due to an internet, proxy problem.
See [Desed repo][desed] or [Desed_website][desed_website] for more info.

### Description
- The **sound event detection** dataset is using [desed] dataset.

#### dataset
The dataset for sound event detection of DCASE2020 task 4 is composed of:
- Train:
	- *weak *(DESED, recorded, 1 578 files)*
	- *unlabel_in_domain *(DESED, recorded, 14 412 files)*
	- synthetic20/soundscapes *(DESED, 2584 files)*
- *Validation (DESED, recorded, 1 168 files):
	- test2018 (288 files)
	- eval2018 (880 files)


#### Baselines dataset
##### SED baseline
- Train:
	- weak
	- unlabel_in_domain
	- synthetic20/soundscapes
- Validation:
	- validation


### Annotation format

#### Weak annotations
The weak annotations have been verified manually for a small subset of the training set. 
The weak annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event_labels (strings)]
```
For example:
```
Y-BJNMHMZDcU_50.000_60.000.wav	Alarm_bell_ringing,Dog
```

#### Strong annotations
Synthetic subset and validation set have strong annotations.

The minimum length for an event is 250ms. The minimum duration of the pause between two events from the same class 
is 150ms. 
When the silence between two consecutive events from the same class was less than 150ms the events have been merged 
to a single event.
The strong annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event onset time in seconds (float)][tab][event offset time in seconds (float)][tab][event_label (strings)]
```
For example:

```
YOTsn73eqbfc_10.000_20.000.wav	0.163	0.665	Alarm_bell_ringing
```


## Authors

|Author                 | Affiliation                |
|-----------------------|---------------             |
|Hao Yen                | National Taiwan University |
|Pin-Jui Ku             | National Taiwan University |

## Contact
If you have any problem feel free to contact Hao Yen (b05901090@ntu.edu.tw) or Pin-Jui Ku (b05901107@ntu.edu.tw)

[dcase2019-task4]: http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments
[dcase-discussions]: https://groups.google.com/forum/#!forum/dcase-discussions
[desed]: https://github.com/turpaultn/DESED
[website]: http://dcase.community/challenge2020/

[scripts]: ./scripts
[GL-MT]: ./GL-MT
