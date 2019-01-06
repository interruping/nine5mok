# Nine5Mok

9 x 9 Go game board piece position recognition Deep-learning model.
This Repository contains pre-trained weights that record with 98% validation accuracy.

## Nine5Mok Model Summary (model.py)

![img/model_summary.png](https://github.com/interruping/nine5mok/raw/master/img/model_summary.png)
 
## Requirment

- Python 3.6.X
- pip3

### install requirments

    >pip3 install -r requirments.txt

## Dataset
- ./data/train50k.csv : consist with index ( filename ) , board columns
- ./data/train50k_s/*.png : input image files ( 120 x 90 ).
	* Image dataset were generated with python script. and using [Active vision dataset](http://cs.unc.edu/~ammirato/active_vision_dataset_website/) as background.

### Input image & Label sample

![img/dataset_sample.png](https://github.com/interruping/nine5mok/raw/master/img/dataset_sample.png)

## Training history

- orange line: training values
- blue line: validation values

![img/training_loss.png](https://github.com/interruping/nine5mok/raw/master/img/training_loss.png =400px)
![img/training_accuracy.png](https://github.com/interruping/nine5mok/raw/master/img/training_accuracy.png =400px)

## Train

    >python train.py
