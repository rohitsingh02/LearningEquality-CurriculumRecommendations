# 18th Place Solution LearningEquality-CurriculumRecommendations

It's 18th place solution to Kaggle competition: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations

This repo contains the code i used to train the models, while the solution writeup is available here: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394910

### HARDWARE: (The following specs were used to create the original solution)

Almost all models were trained on 2xA30 machine.

* OS: Ubuntu 20.04.4 LTS
* CPU: Intel Xeon Gold 5315Y @3.2 GHz, 8 cores
* RAM: 44Gi 
* GPU: 2 x NVIDIA RTX A30 (24 GB)


### SOFTWARE (python packages are detailed separately in `requirements.txt`):

* Python 3.9.13
* CUDA 11.6
* nvidia drivers v510.73.05


### Training (Rohit Part)

* Downlaod additional training data from https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data and extract it to `./data` directory. and from https://www.kaggle.com/datasets/rohitsingh9990/data-retriever and extract it to `./data_retriever` directory.

> Files Description - 
  * `./data/folds.csv` - folds csv file containing topic_id and fold columns.
  * `./data_retriever/train_retriever.csv` - file use to train stage1 models.
  * `./data_retriever/train_ret_mpnet_50_f0_64.csv` to `./data_retriever/train_ret_mpnet_50_f4_64.csv` - files used to train stage2 model.

* To train stage1 models run: `./src/train_s1.sh` this will train and save model weights in `./output/mpnet` directory.
* To generate train data for stage2 model, run `./src/get_s2_candidate.sh` this will create and save files used to train stage2 model [train_ret_mpnet_50_f0_64.csv - train_ret_mpnet_50_f4_64.csv]
* To train stage2 models run: `./src/train_s2.sh` this will train and save model weights in `./model` directory.


Ensemble :
* A simple ensemble of 5 fold mpnet model in both stage1 and stage2 has `CV:0.64 | Public LB:0.657 | Private LB:0.697`
* This simple inference kernel can be accessed here: https://www.kaggle.com/code/rohitsingh9990/fork-of-fork-of-inference-multiple-folds-4bd8ef?scriptVersionId=121960627

### Inference

Final inference kernel with `CV:0.741 | Public LB:0.686 | Private LB:721` is available here: https://www.kaggle.com/code/evgeniimaslov2/fork-of-inference-multiple-folds-yev-with-m-446ea4?scriptVersionId=122117719
