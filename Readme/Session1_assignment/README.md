## Session 1 Assignment - Team Submission
Team Members
1. S.A.Ezhirko
2. Naga Pavan Kumar Kalepu

### Session 1 is about MLOPs Introduction and understanding Version Control.
**Got Good understanding on** : <br />
* Data Pipelines
* Model and Data Versioning
* Model Validation
* Data Validation
* Monitoring

**Trying out DVC** : <br />
DVC is a Data version control. It takes care of 
* Data and model versioning
* Data and model access
* Data pipelines
* Metrics, parameters and plots
* Experiments

### Session 1 Assignment Execution
* Converted a CNN model which was developed using Keras into PyTorch version. Find Ezhirko implementation [here](https://github.com/emlo1convergence/emlo1/blob/main/Ezhirko/Pytorch_train.py) and Pavan implementation [here](https://github.com/emlo1convergence/emlo1/blob/main/Pavan/train.py)
* Installed DVC
* Cloned a Git repository which had [tutorial files](https://github.com/iterative/example-versioning.git) to our local drive.
* Created a virtual environment on that drive and installed the required [python libraries](https://github.com/emlo1convergence/emlo1/blob/main/Ezhirko/requirements.txt)
* Initial set of data.zip which contains cats and dogs images that was used for trainning was captured using dvc get.
* Current state of initial dataset was caaptured using dvc add data
* The pytorch model was trained for 11 epoch and achieved Test accuracy of 96.72 link to Ezhirko [metrics file](https://github.com/emlo1convergence/emlo1/blob/main/Ezhirko/Metrics.csv) and Pavan [metrics file](https://github.com/emlo1convergence/emlo1/blob/main/Pavan/Metrics.csv)
* Ezhirko [data.dvc,best model.dvc,metrics.csv and gitignore](https://github.com/emlo1convergence/emlo1/tree/main/Ezhirko) and Pavan [data.dvc,best model.dvc,metrics.csv and gitignore](https://github.com/emlo1convergence/emlo1/tree/main/Pavan) was commited and push to our git repository
* Second Version of data was downloaded and the additional images were merged to our original data.
* The current state of merged data was again captured using dvc add data
* The pytorch model was re trained for 11 epoch and achieved Test accuracy of 94.40 link to Ezhirko [metrics file](https://github.com/emlo1convergence/emlo1/blob/main/Ezhirko/Metrics.csv) and Pavan [metrics file](https://github.com/emlo1convergence/emlo1/blob/main/Pavan/Metrics.csv)
* Best model dvc was captured. Ezhirko [data.dvc,best model.dvc,metrics.csv and gitignore](https://github.com/emlo1convergence/emlo1/tree/main/Ezhirko) and Pavan [data.dvc,best model.dvc,metrics.csv and gitignore](https://github.com/emlo1convergence/emlo1/tree/main/Pavan) was commited and push to our git repository
* Pytest unit test cases was written in the file [test_Push.py](https://github.com/emlo1convergence/emlo1/blob/main/Ezhirko/test_Push.py)
  *This file had test case written to verify if we have NOT uploaded data.zip
  *To verify if we have NOT uploaded model.h5 file
  *To verify the accuracy of the model is more than 70%
  *To verify the accuracy of your model for cat and dog is independently more than 70%
 * Python workflow pipe line was created using Git Actions and pytest execution task was added to execute the test case on push job. link to [workflow file](https://github.com/emlo1convergence/emlo1/blob/main/.github/workflows/python-app.yml)
