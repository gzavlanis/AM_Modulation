# AM_Modulation
 Prediction of an Amplitude Modulated (AM) signal using a neural network

In this project, the code asks the user to enter the specifications in order to create an amplitude modulated signal. 
Then, 10dB of noise are added to this signal. The resulting signal is used to train a BiLSTM deep learning neural network. 
The neural network is then called to predict the signal. The results are amazing! Finally, the program asks the user to re-enter specifications for a new signal. 
The signal is generated and 5dB of noise are added. The neural network is called upon to predict the signal, without prior training on it. 
It turns out that the neural network is able, after its training, to predict with great accuracy any signal with AM modulation. 
