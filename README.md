# VaryingSkinTone

This repository contains the Keras implementation of the paper titled <br>
"**An Unsupervised Approach towards Varying Human Skin Tone Using Generative Adversarial Networks**"
https://ieeexplore.ieee.org/abstract/document/9412852

This is an inference code. 
The checkpoints and a sample dataset are upload at,
https://drive.google.com/drive/folders/1eVO9ki1drp1fGjrucNgd24iL1A6MB_0Y?usp=sharing

Testing:
=============================================================================================================
This code can be run in 2 ways: 
1. Using our trained segmentation model the skin segmentation part
    To execute in this case:<br>
    set the following in the params.py file <br>
        dataset_path = './Datasets/DeepFashion/Category-and-Attribute-Prediction-Benchmark/'<br>
    Command to run:<br>
        python2 stage/test_skin.py -batch_size 1 -range_count 10 -seg_choice False -test_filename deepfashion_names_upper.txt<br>

3. Using pre-computed segmentations by some other method of your choice and then running our skin tone changing model on the provided image and the estimated segmentation.<br>
    To execute in this case:<br>
    set the following in the params.py file <br>
        dataset_path = './Datasets/MyData/'<br>
    Command to run:<br>
        python2 stage/test_skin.py -batch_size 1 -range_count 10 -seg_choice True -test_filename test_external.txt<br>


--------------------------------------------------------------------------------------------------------------
The code is tested in the following versions:<br>
keras = '2.2.4'
tensorflow = '1.14.0'
python = '2.7.12'


In case you use this code please consider citing

      @inproceedings{roy2021unsupervised,
      title={An Unsupervised Approach towards Varying Human Skin Tone Using Generative Adversarial Networks},
      author={Roy, Debapriya and Mukherjee, Diganta and Chanda, Bhabatosh},
      booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
      pages={10681--10688},
      year={2021},
      organization={IEEE}}
