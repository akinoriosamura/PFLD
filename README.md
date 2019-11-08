# PFLD

## requirements
 - download WFLW images and annotations  
 [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)

 - untar and set WFLW in `/data/`  
 - get `model.meta` and set in `models2/models/`  
 [Google drive](https://drive.google.com/open?id=1Ol-JWNS3bVTD3hV6aIbTm6cNzGOWiw2U)  

## If use GPU
  - install docker and docker-compose
  ```
  docker-compose up -d
  docker attach tensorflow-gpu-1-14
  ```
 
## create pipenv 
 - install pipenv  

 - install library and shell  
 ```
 pipenv install
 pipenv shell
 ```

## preprocess WFLW
```
cp euler_angles_utils.py ./data/
cd data
python SetPreparation.py WFLW [label num]
```

## train
```
sh train.sh
```

## test
```
python test_model.py
```

## camera test
```
python camera.py
```

## Run by CPU ex
```
CUDA_VISIBLE_DEVICES= python test_model.py
```

## save model
### to saved_model and pb file
 - fix save.sh file  
 - run `save.sh`

### to tflite file
 - from pb
```
tflite_convert --output_file=[tflite path] --graph_def_file=[pb file path] --input_arrays=image_batch --output_arrays=pfld_inference/fc/BiasAdd --allow_custom_ops --enable_select_tf_ops
```

 - from SavedModel
```
fix convert_tflite.py
python convert_tflite.py
```


## original README
Table 1:  
  The code for cauculating euler angles prediction loss have been released.

Table 2:     
  I`ve done the job and fixed the memory leak bug:
  The code has a flaw that i calculate euler angles ground-truth while training process,so the training speed have slowed down because  some work have to be finished on the cpu ,you should calculate the euler angles in the preprocess code    
      
Table 3：It is an open surce program reference to https://arxiv.org/pdf/1902.10859.pdf , if you find any bugs or anything incorrect,you can notice it in the issues and pull request,be glad to receive you advices.     
And thanks @lucknote for helping fixing existing bugs.
  
EASY TO TRAIN:
>STEP1:data/SetPreparation.py  
>STEP2:train.sh
  
  
SAMPLE IMGS:  

 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/10.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/121.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/17.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/19.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/21.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/52.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/7.jpg)
        
 SAMPLE VIDEO:  

 ![Image text](data/sample_imgs/ucgif_20190809185908.gif)
