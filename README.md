# neural-transliterate
Exploring Bulgarian to English transliteration for Computational Linguistics 1 (CMSC 723) at UMD


## How to run

### Runing the model

Run the default model:
```
python transliterate.py -t data/en_bg.train.txt -v data/en_bg.val.txt -n 20000 -o data/en_bg.val.out
```

Without attention mechanism
```
python noattention.py -t data/en_bg.train.txt -v data/en_bg.val.txt -n 20000 -o data/en_bg.val.out
```

With beamsearch decoding
```
python beamDecoder.py -t data/en_bg.train.txt -v data/en_bg.val.txt -n 20000 -o data/en_bg.val.out  
```

### Runing the experiments

To run the eperiments described in the report, simply run the assosiated python script. 
Each of them is named with the perfix 'ex_'.
