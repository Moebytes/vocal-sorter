# Vocal Sorter

The vocal sorter aims to speed up sorting tasks of filtering out only 
the good chops obtained from running the plugin Vocal Chopper:

https://github.com/Moebytes/Vocal-Chopper

It's a CLAP based model that is trained on two folders "good" and "bad" in the 
data directory, containing your specific vocal preferences. Once the model is trained, it does 
a fairly good job at removing all the vocals with undesired qualities (eg. mostly silence, 
unclear, too much background noise). 

After cloning the project make sure to add your vocals into the data directory:

```
data
├── good
└── bad
```

### Usage

Train:
```
python3 train_sorter.py
```

Inference:
```
python3 sort_chops.py "path/to/folder"
```

### Pretrained model

https://huggingface.co/Moebits/vocal-quality/tree/main
