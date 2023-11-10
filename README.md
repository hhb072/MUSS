# MUSS
Code for ["Memory Uncertainty Learning for Real-World Single Image Deraining" with a real-world rain dataset"](https://ieeexplore.ieee.org/document/9789487) 

## Training

Please download the real-world train subset and test set of SSID in the [link](https://drive.google.com/drive/folders/1bSOX_lSnuTDkwpOXw17up7zEPaqFXeVg?usp=sharing) as well as the [DDN](https://drive.google.com/file/d/10cu6MA4fQ2Dz16zfzyrQWDDhOWhRdhpq/view?usp=sharing) dataset and [SPA](https://stevewongv.github.io/derain-project.html) dataset and put them in the data folder. Adjust the parameters in  'config.yaml'  according to your own settings. 

Train MUSS:

	 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config_file='train.yaml'  

Test SWAL:

    python test.py --config_file='test.yaml' 

## Citation

If you use our codes, please cite the following paper:

	 @inproceedings{huang2022memory,
	   title={Memory uncertainty learning for real-world single image deraining},
	   author={Huang, Huaibo and Luo, Mandi and He, Ran},
	   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	   volume={45},
	   number={3},
	   pages={3446--3460},
	   year={2022},
	   publisher={IEEE}
	  }
 
**The released codes are only allowed for non-commercial use.**

