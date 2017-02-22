
# Hidden Markov Model (HMM)
This repository contains a from-scratch Hidden Markov Model implementation utilizing the Forward-Backward algorithm 
and Expectation-Maximization for probabilities optimization. Please note that this code is not yet optimized for large 
sequences. More specifically, with a large sequence, expect to encounter problems with computational underflow. This will be 
resolved in the next release. 

## Methodology

Hidden Markov models are used to ferret out the underlying, or hidden, sequence of states that generates a set of observations. In his now canonical toy example, [Jason Eisner](http://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp) uses a series of daily ice cream consumption (1, 2, 3) to understand Baltimore's weather for a given summer (Hot/Cold days). These are arrived at using transmission probabilities (i.e. the likelihood of moving from one state to another) and emission probabilities (i.e. the likelihood of seeing a particular observation given an underlying state).  

This implementation adopts his approach into a system that can take: 

- An initial transmission matrix
- An initial emission matrix
- A set of observations

You can see an example input by using the `main()` function call on the `hmm.py` file. 

HMM models calculate first the probability of a given sequence and its individual observations for possible hidden state sequences, then re-calculate the matrices above given those probabilities. By iterating back and forth (what's called an expectation-maximization process), the model arrives at a local optimum for the tranmission and emission probabilities. It's a pretty good outcome for what might otherwise be a very hefty computationally difficult problem. 

This model implements the [forward-backward](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) algorithm recursively for probability calculation within the broader [expectation-maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) pattern.

## Use of Current Release

This system can currently: 

- Train an HMM model on a set of observations, given a number of hidden states N
- Determine the likelihood of a new set of observations given the training observations and the learned hidden state probabilities

## To Come

I'm a full time student and this is a side project. It's still in progress. Things to come:

 - Further methodology
 - Full Testing suite
 - Viterbi decoding for understanding the most likely sequence of hidden states

## Sample Usage 

`emission = np.array([[0.7, 0], [0.2, 0.3], [0.1, 0.7]])`  
`transmission = np.array([ [0, 0, 0, 0], [0.5, 0.8, 0.2, 0], [0.5, 0.1, 0.7, 0], [0, 0.1, 0.1, 0]])`  
`observations = ['2','3','3','2','3','2','3','2','2','3','1','3','3','1','1',`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`'1','2','1','1','1','3','1','2','1','1','1','2','3','3','2',`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`'3','2','2']`  
`model = HMM(transmission, emission)`  
`model.train(observations)`  
`new_seq = ['1', '2', '3']`  
`likelihood = model.likelihood(new_seq)`
