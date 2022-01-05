import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, LeakyReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf
import random
from keras.utils.vis_utils import plot_model
import matplotlib.patheffects as path_effects

A_c= float(input('Enter carrier amplitude: '))
f_c= float(input('Enter carrier frequency: '))
A_m= float(input('Enter message amplitude: '))
f_m= float(input('Enter message frequency: '))
modulation_index= float(input('Enter modulation index: '))

# create the modulated signal
t= np.linspace(0, 1, 3000)
carrier= A_c* np.cos(2* np.pi* f_c* t)
modulator= A_m* np.cos(2* np.pi* f_m* t)
product= A_c* (1+ modulation_index* np.cos(2* np.pi* f_m* t))* np.cos(2* np.pi* f_c* t)
print(product)

plt.subplot(3, 1, 1)
plt.title('Amplitude Modulation (AM)')
plt.plot(t, modulator, 'g')
plt.ylabel('Amplitude')
plt.xlabel('Message signal')
plt.subplot(3, 1, 2)
plt.plot(t, carrier, 'r')
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')
plt.subplot(3, 1, 3)
plt.plot(t, product, color= 'purple')
plt.ylabel('Amplitude')
plt.xlabel('AM signal')
plt.subplots_adjust(hspace= 1)
plt.rc('font', size= 15)
plt.show()
fig= plt.gcf()
fig.set_size_inches(16, 9)

x_volts= product # modulated signal
x_watts= x_volts** 2 # Signal power in Watts
x_db= 10* np.log(x_watts) # Signal power in dB
plt.subplot(3, 1, 1)
plt.plot(t, x_volts)
plt.title('Signal')
plt.ylabel('Voltage (Volts)')
plt.xlabel('Time (nanoseconds)')
plt.subplot(3, 1, 2)
plt.plot(t, x_watts)
plt.title('Signal Power')
plt.ylabel('Power (Watts)')
plt.xlabel('Time (nanoseconds)')
plt.subplot(3, 1, 3)
plt.plot(t, x_db)
plt.title('Signal Power in dB')
plt.ylabel('Power (dB)')
plt.xlabel('Time (nanoseconds)')
plt.subplots_adjust(hspace= 1)
plt.rc('font', size= 15)
plt.show()
fig1= plt.gcf()
fig1.set_size_inches(16, 9)

# Adding noise using a target noise power
target_noise_db= 10 # Set a target channel noise power to something very noisy
target_noise_watts= 10** (target_noise_db/ 10) # Convert to linear Watt units
mean_noise= 0 # Generate noise samples
noise_volts= np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(x_watts))
y_volts= x_volts+ noise_volts # Noise up the original signal (again) and plot
print(y_volts)

# Plot signal with noise
plt.subplot(2, 1, 1)
plt.plot(t, y_volts)
plt.title('Signal with noise')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
y_watts= y_volts** 2
y_db= 10* np.log10(y_watts)
plt.subplot(2, 1, 2)
plt.plot(t, 10* np.log10(y_volts** 2))
plt.title('Signal with noise')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.subplots_adjust(hspace= 1)
plt.rc('font', size= 15)
plt.grid()
plt.show()

n_timesteps= 3 #define the timesteps of the problem
# Create the deep BiLSTM network and make a figure of it
model= Sequential()
model.add(Bidirectional(LSTM(32, return_sequences= True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(128, activation= LeakyReLU(alpha= 0.01))))
model.add(Dense(128, activation= LeakyReLU(alpha= 0.01)))
model.add(Dense(64, activation= LeakyReLU(alpha= 0.01)))
model.add(Dense(1, activation= 'linear'))
model.compile(loss= MeanSquaredError(), optimizer= Adam(learning_rate=0.01), metrics=[ 'accuracy' ])
print(model.summary())
plot_model(model, to_file='BiLSTM.png', show_shapes=True, show_layer_names=True)

X= np.array([x_volts]) # input training signal
Y= np.array([y_volts]) # output training signal
print(X)
print(Y)

# convert the arrays to tensors
X= tf.convert_to_tensor(X, dtype= tf.float32)
Y= tf.convert_to_tensor(Y, dtype= tf.float32)
print(X)
print(Y)

# reshape data for entering the model
X= tf.reshape(X, [1000, n_timesteps, 1])
Y= tf.reshape(Y, [1000, n_timesteps, 1])
print(X)
print(Y)

# train the model and plot results for loss
def train_model(model, n_timesteps):
    hist= model.fit(X, Y, epochs= 20, batch_size= 64)
    loss= hist.history['loss']
    return loss

loss= train_model(model, n_timesteps)
plt.plot(loss, label= 'Loss')
plt.title('Training loss of the model')
plt.xlabel('epochs', fontsize= 18)
plt.ylabel('loss', fontsize= 18)
plt.grid()
plt.legend()
plt.show()

# predict the output signal Y
Yhat= model.predict(Y, verbose= 0)
Yhat= Yhat.reshape(3000, 1)
print(Yhat)

# predict the signal without noise X
Xhat= model.predict(X, verbose= 0)
Xhat= Xhat.reshape(3000, 1)
print(Xhat)

# Reshape X and Y back to normal shape
X= tf.reshape(X, [3000, 1])
Y= tf.reshape(Y, [3000, 1])

# Plot real waveform and predicted waveform for Y
plt.plot(t, Y, 'r', label= 'Real waveform')
plt.plot(t, Yhat, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

# Plot real waveform and predicted waveform for X
plt.plot(t, X, 'r', label= 'Real waveform')
plt.plot(t, Xhat, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

print('Create a new waveform with parameters asked from the user and try to predict it with the previous training')

A_c2= float(input('Enter carrier amplitude: '))
f_c2= float(input('Enter carrier frequency: '))
A_m2= float(input('Enter message amplitude: '))
f_m2= float(input('Enter message frequency: '))
modulation_index2= float(input('Enter modulation index: '))

# create the modulated signal
t2= np.linspace(0, 1, 3000)
carrier2= A_c2* np.cos(2* np.pi* f_c2* t2)
modulator2= A_m2* np.cos(2* np.pi* f_m2* t2)
product2= A_c2* (1+ modulation_index2* np.cos(2* np.pi* f_m2* t2))* np.cos(2* np.pi* f_c2* t2)
print(product2)

x_volts2= product2 # modulated signal
x_watts2= x_volts2** 2 # Signal power in Watts
x_db2= 10* np.log(x_watts2) # Signal power in dB

# Adding noise using a target noise power
target_noise_db2= 5 # noise power
target_noise_watts2= 10** (target_noise_db2/ 10) # Convert to linear Watt units
mean_noise2= 0 # Generate noise samples
noise_volts2= np.random.normal(mean_noise2, np.sqrt(target_noise_watts2), len(x_watts2))
y_volts2= x_volts2+ noise_volts2 # Noise up the original signal (again) and plot
print(y_volts2)

# Plot signal with noise
plt.subplot(2, 1, 1)
plt.plot(t2, y_volts2)
plt.title('Signal with noise')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
y_watts2= y_volts2** 2
y_db2= 10* np.log10(y_watts2)
plt.subplot(2, 1, 2)
plt.plot(t2, 10* np.log10(y_volts2** 2))
plt.title('Signal with noise')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.subplots_adjust(hspace= 1)
plt.rc('font', size= 15)
plt.grid()
plt.show()

X2= np.array([x_volts2]) # input training signal
Y2= np.array([y_volts2]) # output training signal
print(X2)
print(Y2)

# convert the arrays to tensors
X2= tf.convert_to_tensor(X2, dtype= tf.float32)
Y2= tf.convert_to_tensor(Y2, dtype= tf.float32)
print(X2)
print(Y2)

# reshape data for entering the model
X2= tf.reshape(X2, [1000, n_timesteps, 1])
Y2= tf.reshape(Y2, [1000, n_timesteps, 1])
print(X2)
print(Y2)

# predict the output signal Y
Yhat2= model.predict(Y2, verbose= 0)
Yhat2= Yhat2.reshape(3000, 1)
print(Yhat2)

# predict the signal without noise X
Xhat2= model.predict(X2, verbose= 0)
Xhat2= Xhat2.reshape(3000, 1)
print(Xhat2)

# Reshape X and Y back to normal shape
X2= tf.reshape(X2, [3000, 1])
Y2= tf.reshape(Y2, [3000, 1])

# Plot real waveform and predicted waveform for Y
plt.plot(t2, Y2, 'r', label= 'Real waveform')
plt.plot(t2, Yhat2, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

# Plot real waveform and predicted waveform for X
plt.plot(t2, X2, 'r', label= 'Real waveform')
plt.plot(t2, Xhat2, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(5, 1.5))
text = fig.text(0.5, 0.5, 'You can change the values and\n' 'experiment as long as you want!' ,ha='center', va='center', size=20)
text.set_path_effects([path_effects.Normal()])
plt.show()
