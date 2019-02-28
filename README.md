# ECG Classification

ECG Classification based on MLP RNN LSTM Attention-Model CNN
## Materials

* MIT Arrythmia database and MIT Normal Sunis Database
* CCDD database

## Introduce
* The DeHaze folder is a dehaze model of image
* EEG folder is a EEG classification model
* other ECG model folder contains some simple models or some ideas for trying
* 12-Lead ECG model is four deep learning model which build with pytorch
  * Vanilla-CNN is a simple CNN model to classify the CCDD database
  * Channel-RNN is a CNN+RNN network
  * Featrue-CNN is a RNN+CNN network
  * Multi-RNN is a 12-Lead based RNN network

## Conclusion
ECG signals were classified using different deep learning models. And try to combine LSTM with CNN to process multi-lead sequence signals.
The model performance is not particularly good, but I hope these idea will help you a little.
