# RNN for Text Prediction 

This repository contains a very basic implementation of a Recurrent Neural Network (RNN) using NumPy. It's designed as a learning example to demonstrate the core concepts of RNNs, forward propagation, backward propagation (Backpropagation Through Time - BPTT), and training with gradient descent on a small, fixed text sequence.

## Overview

The script builds a simple RNN model trained to predict the next word in a specific phrase. It uses a minimal vocabulary derived from the phrase itself and trains the network over several epochs to minimize the error in predicting the target word.

The specific task hardcoded in the script is predicting the word "best" given the input sequence "barca is the".

## How it Works

1.  **Data Preparation:**
    *   A small list of words (`text = ["barca", "is", "the", "best"]`) defines the training data.
    *   A vocabulary is created from these words.
    *   Dictionaries are made to map words to integer indices and vice-versa.
    *   The input text is converted into a sequence of indices.
2.  **Model Initialization:**
    *   The RNN has an input layer, a hidden layer, and an output layer.
    *   The size of the input and output layers is determined by the vocabulary size.
    *   A fixed `hidden_size` is chosen (10 in this case).
    *   Weight matrices (`Wxh`, `Whh`, `Why`) and bias vectors (`bh`, `by`) are initialized with small random values or zeros.
3.  **Forward Propagation:**
    *   The input sequence ("barca is the") is processed one word at a time.
    *   Each word is converted into a one-hot vector.
    *   The network calculates the hidden state at each timestep using the current input and the previous hidden state. A `tanh` activation is used for the hidden layer.
    *   The output layer calculates scores for each word in the vocabulary based on the current hidden state.
    *   A `softmax` function converts these scores into probability distributions over the vocabulary.
4.  **Loss Calculation:**
    *   During training, the loss is calculated at the end of the input sequence (after processing "the").
    *   The loss is the negative log-likelihood of the actual target word ("best").
5.  **Backward Propagation (BPTT):**
    *   Gradients are calculated starting from the output layer at the last timestep and propagated backward through time and through the network layers.
    *   This determines how much each weight and bias contributed to the error.
    *   Gradient clipping is applied to prevent exploding gradients.
6.  **Weight Update:**
    *   Weights and biases are updated using the calculated gradients and a specified `learning_rate`. This is the optimization step that allows the network to learn.
7.  **Training Loop:**
    *   The forward and backward propagation steps are repeated for a fixed number of `epochs`.
    *   The hidden state from the end of one epoch's processing is carried over to the start of the next epoch for the *same* input sequence.
8.  **Prediction:**
    *   After training, the network is run through the input sequence ("barca is the") one last time using the learned weights.
    *   The word with the highest probability at the last timestep's output is selected as the prediction.

## Requirements

*   Python 3.x
*   NumPy library

## Installation

1.  Clone this repository (or copy the code into a Python file).
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Install the required library:
    ```bash
    pip install numpy
    ```

## Usage

1.  Save the provided Python code as a `.py` file (e.g., `simple_rnn.py`).
2.  Run the script from your terminal:
    ```bash
    python simple_rnn.py
    ```

## Expected Output

*   You will see the loss printed every 100 epochs during training, followed by the final predicted word:
*   Starting Training...
*   Epoch 0, Loss: ...
*   Epoch 100, Loss: ...
*   Epoch 200, Loss: ...
*   ...
*   Epoch 900, Loss: ...
*   Training Finished.
*   Making Prediction...
*   Input sequence: barca is the best
*   Predicted word: best
