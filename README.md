# SegFormers@DisRPT2021

Repository for our systems submitted to the DisRPT 2021 Shared Task for Discourse Unit Segmentation, Connective Detection and Discourse Relation Classification.

## Directory Structure

**code** folder contains the source code for the systems submitted by us. <br>
**data** folder contains the official datasets for all 16 languages provided by the organizers. (Had to be removed as we cannot put the restored text publicly online for the licensed datasets  <br>
**utils** folder contains the official scorer provided by the organizers. <br>

## Code for the baseline model

**code/LSTM** contains the code for our baseline bidirectional LSTM model.

**Prerequisite packages and modules -** 

* python (version 3.7.4)
* torch (version 1.9.0)
* numpy
* sklearn
* tqdm, os, sys, io, enum

Install all packages using the `pip3 install` command

**Usage -** 

The model is run using the command `python3 train_baseline.py [dataset name]`.  <br>
The dataset name has to be provided in the format `LANG.FRAMEWORK.CORPUS`, e.g. `eng.rst.gum`.  <br>
The model stores the output prediction files in the **./outputs** folder. <br>
Scores are obtained by running the official scorer `seg_eval.py` from the utils directory.  <br>

e.g. -
```
# Training the baseline model on a dataset

python3 train_baseline.py deu.rst.pcc

# Obtaining the scores on the dev set

python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_dev.tok ./outputs/deu.rst.pcc_dev.preds

# Obtaining the scores on the test set

python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_test.tok ./outputs/deu.rst.pcc_test.preds
```

Call the above python files separately for each of the 15 available datasets to get the reported baseline results.

Some of the datasets (`eng.pdtb.pdtb`, `rus.rst.rrt`, `tur.pdtb.pdtb`) are very large in size due to which the training of the model can take a lot of time. <br>
We have provided a `batch.sh` file which can be used with the SLURM workload manager's `sbatch` command if SLURM is available. <br> 
`sbatch batch.sh` will run the `train_baseline.py` file on the `eng.pdtb.pdtb`, `rus.rst.rrt`, `tur.pdtb.pdtb` whose outputs will be stored in the ./**outputs** folder which can then be evaluated using `seg_eval.py` in the same way as mentioned above. <br> 
You can edit `batch.sh` to add more datasets if needed.

## Code for SegFormers

**code/Transformers** contains the code for our main model, SegFormers.

**Prerequisite packages and modules -**

* python (version 3.7.4)
* torch (version 1.9.0)
* transformers
* numpy
* sklearn
* tqdm, os, sys, io, enum

Install all packages using the `pip3 install` command. <br>

**Usage -**

For training the model on the **.conllu** files - <br> 
The model is run using the command `python3 train_final_conllu.py [dataset name]`. <br> 
The dataset name has to be provided in the format `LANG.FRAMEWORK.CORPUS`, e.g. `eng.rst.gum`. <br> 
The model stores the output prediction files in the **./outputs** folder. <br> 
Scores are obtained by running the official scorer `seg_eval.py` from the utils directory.

For training the model on the **.tok** files - <br>
The model is run using the command `python3 train_final_tok.py [dataset name]`. <br> 
The dataset name has to be provided in the format `LANG.FRAMEWORK.CORPUS`, e.g. `eng.rst.gum`. <br> 
The model stores the output prediction files in the **./outputs** folder. <br> 
Scores are obtained by running the official scorer `seg_eval.py` from the utils directory. 

e.g. - 
* For the **.conllu** files -
    ```
    # Training SegFormers on .conllu files

    python3 train_final_conllu.py deu.rst.pcc

    # Obtaining the scores on the dev set

    python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_dev.conllu ./outputs/deu.rst.pcc_dev.conllu.preds

    # Obtaining the scores on the test set

    python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_test.conllu ./outputs/deu.rst.pcc_test.conllu.preds
    ```

* For the **.tok** files - 
    ```
    # Training SegFormers on .tok files

    python3 train_final_tok.py deu.rst.pcc

    # Obtaining the scores on the dev set

    python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_dev.tok ./outputs/deu.rst.pcc_dev.tok.preds

    # Obtaining the scores on the test set

    python3 ../../utils/seg_eval.py ../../data/deu.rst.pcc/deu.rst.pcc_test.tok ./outputs/deu.rst.pcc_test.tok.preds
    ```

Call the above python files separately for each of the 15 available datasets to get the reported final results.

All our final results have been obtained after training the model on our institute GPU cluster. <br> 
Training with `cuda` is recommended.

Some of the datasets (`eng.pdtb.pdtb`, `rus.rst.rrt`, `tur.pdtb.pdtb`) are very large in size due to which the training of the model can take a lot of time. <br> 
We have provided a `batch.sh` file which can be used with the SLURM workload manager's `sbatch` command if SLURM is available. <br> 
`sbatch batch.sh` will run the `train_final_conllu.py` and `train_final_tok.py` files on the `eng.pdtb.pdtb`, `rus.rst.rrt`, `tur.pdtb.pdtb` whose outputs will be stored in the ./**outputs** folder which can then be evaluated using `seg_eval.py` in the same way as mentioned above. <br> 
You can edit `batch.sh` to add more datasets if needed.

## Data

**data** folder contains the original data provided by the organizers. <br> 
The Chinese PDTB dataset does not contain text since we could not obtain the original Chinese PDTB dataset to reconstruct the provided files.

## Utils

**utils** folder contains the official scorer provided by the organizers for the 2 tasks. <br> 
All final scores have been calculated using seg_eval.py.
