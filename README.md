# PFLD

## requirements
 - download WFLW images and annotations  
 [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)

 - untar and set WFLW in `/data/`  
 - get `model.meta` and set in `models2/models/`  
 [Google drive](https://drive.google.com/open?id=1Ol-JWNS3bVTD3hV6aIbTm6cNzGOWiw2U)  

## prepare env
  - install docker and docker-compose
  ```
  docker-compose -f docker-compose.yml up -d
  docker attach [container name]
  ```
  - install
  ```
  pip install -r requirement.txt
  ```

if you use pfld tensorflow 2 ver, you change above script like `docker-compose.yml` to `docker-compose2.yml` and `requirement.txt` to `requirement2.txt`.

## preprocess dataset
```
cd data
```
 - set dataset config
 add processing dataset's config to `preparate_config.ini`.
 ```ex
 [`datasetname`_`label_num`]
 imageDirs = ./growing/growing_20180601
 Mirror_file = ./Mirror68.txt
 landmarkTrainDir = ./growing/traindata8979_20180601_train.txt
 landmarkTestDir = ./growing/traindata8979_20180601_test.txt
 landmarkTestName = traindata8979_20180601_test.txt
 outTrainDir = train_growing52_data
 outTestDir = test_growing52_data
 ImageSize = 112
 ```
 - set augment scale
 update AUGMENT_NUM in `SetPreparation.py`
 - run
 ```
 python SetPreparation.py `datasetname` `label_num` `rotate or nonrotate` `hard aug or nothard aug`
 ```
 ```ex
 python SetPreparation.py pcnWFLW 68 rotate hard
 ```
then, you can get train and test dataset

## run
### some model types
We have some model types and each model has each `run.sh` script.
 - `run.sh` is for `pfld`
 - `run_tf2.sh` is for `pfld tf ver2`
 - `run_tfjs.sh` is for `face api model`
 - `run_xin.sh` is for `xining model`

and below README explain for `run.sh`

### change settings
update `run.sh` and change dataset path and model path and so on settings.

### train
```
sh run.sh train
```

### pretrain
```
sh run.sh pretrain
```

### test
```
sh run.sh test
```

### save tflite
```
sh run.sh save
```
