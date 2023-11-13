Download the ESC V0.9 data and repalce your data path in the EventStoryLine_process.py

Run the following command to get the training data----train.npy
```
python EventStoryLine_process.py
```
Run the following command to get the training model.
````
python main.py --flod 1 --t_lr 1e-5 --sample_rate 0.2
````
And you will get the traing model.
Replace the model path in preidct.py and run the following command.
```
python predict.py --fold 1
```
Running the above command will result in two prediction files.
Replace the test-prediction file path in the discrimination.py and run the following command.
```
python discrimination.py --fold 1
```