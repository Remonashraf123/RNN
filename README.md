RNN for Text Prediction
Overview
This project implements a simple Recurrent Neural Network (RNN) for text prediction. The network is trained to predict the next word in a sequence of words. It uses the Tanh activation function and the softmax function for output prediction. The model is trained using backpropagation through time (BPTT) and gradient descent.

Features
RNN Architecture: A simple RNN with a hidden layer and output layer.

Text Prediction: The network predicts the next word in a sequence.

Training: Trained using gradient descent with backpropagation.

Softmax Output: Uses the softmax function for word prediction.

Dependencies
Make sure you have Python and the following libraries installed:

bash
Copy
Edit
pip install numpy
How It Works
Data Preparation:

Text is tokenized into words.

Each word is converted into an integer index.

A sequence of words is used to predict the next word in the sequence.

Network Architecture:

The network has an input layer representing the vocabulary size.

A hidden layer with a configurable size.

The output layer predicts the next word in the sequence, which corresponds to the vocabulary size.

Forward Propagation:

The network processes a sequence of word indices.

The hidden state is updated at each timestep using the Tanh activation function.

The output at each timestep is computed using the softmax function to predict the next word.

Backward Propagation:

The model computes the gradients using backpropagation and adjusts the weights using gradient descent.

The gradients are clipped to avoid exploding gradients.

Training:

The model is trained for a specified number of epochs, and the loss is computed at regular intervals.

Prediction:

After training, the model predicts the next word in the sequence based on the learned weights.

Code Structure
one_hot(idx, size): Converts word index into a one-hot vector.

softmax(x): Computes the softmax of the output.

forward_propagation(inputs, h_prev): Performs the forward pass through the network.

backward_propagation(xs, hs, ps, targets): Computes the gradients for backpropagation.

Training Loop: The model is trained using gradient descent over a specified number of epochs.

Example Output
yaml
Copy
Edit
Epoch 0, Loss: 1.1234
Epoch 100, Loss: 0.9876
Epoch 200, Loss: 0.7890
...
Predicted word: best
How to Run
Copy the provided code into a Python script file (e.g., rnn_text_prediction.py).

Run the script:

bash
Copy
Edit
python rnn_text_prediction.py
The model will be trained, and the predicted word will be displayed after the training.
