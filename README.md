# COS 700 Research Project

## Generating the Data (Corpus)
We generate the dataset using all the files in the `corpus` directory. I'm not expecting this to change really so the notes are not going to be too in depth, but basically, in order:
- `generate_board.py` goes and generates a board with some mines, calculates the 0-8 value on each cell
- `board_hider.py` then finds a random "safe" start (like simulating the first click to start the game) and hides ALL other cells, leaving only a small portion revealed
- `revealer.py` reveals the cells when a certain cell has been selected
- `generate_dataset.py` generates a whole bunch of boards and stores them in the `cos700_dataset.npz` file
- `extract_sliding_window.py` takes a board, a cell row and column, and a radius, and returns an X*X sliding window centered around that cell

You don't need to do this, it takes a while. I already did all of this and we can just use the dataset files which have been uploaded to git and is in the root directory.

## The Statistics
All in the `stats` folder. These are used mostly for probabilities and calculating the theoretical maximum performance our agents can achieve. Again, not really something we need to know for our implementations, it's more useful just to determine which sliding windows are the most optimal. 

Here they are:
- `brier.py` simply calculates MSE, it's just a fancy name for when we do predictions like this. There are two variants of the brier function or using numpy and not using numpy.
- `mrmr.py` minimum redundancy maximum relevance, the code that determines which cell is the next best to include in our sliding window
- `mututal_information.py` I really tried to understand this one, but the math was hard and I ended up using sklearn to do it for me. Also part of the sliding window analysis
- `shannon.py` calculates shannon entropy used to measure uncertainty in the agents and is part of the code to determine how accurate the agents can be expected to be

## CSP Solver
In the `solver` directory is our **Constraint Satisfaction Problem** solver. This is the guy that goes and calculates the probability of each frontier cell (which is a hidden cell on the edge between the revealed/hidden sections) being a mine for our dataset. All it does really is go through every possible legal possible mine placement, and it will be like "okay so in cell (3,4) there were mines 19 times out of the 38 possible board combinations that satisft all the requirements so the possibility of there being a mine is 19/38=50%". This is REALLY slow especially when there are hundreds of different combinations though.

## The Tools
These are a bit messy, some are just experiments and others are part of other code. AI did all the visualizers here, just play around with them. Coolest are the dataset visualizer and the theoretical maximum visualizer, they are hopefully self explanatory.

`frontier_cells.py`, `constraint_builder.py`, `radial_average.py` and `theoretical_maximum.py` are used elsewhere and not really relevant to this I think.

## The Neural Network
So in the `simon` directory I have my `neural_network.py`. This was designed for my use case so it does not have aby sort of training built into it. Really all it does is the feedforward pass. You initialize it with the hidden layer array, \[3,4\] means a hidden layer of 3 neurons followed by a hidden layer of 4. There is always only one output neuron. The other initialization parameter is simply the number of input features. This depends on what you feed it. For example, if you keep the original mine values from 0-11 then we have 24*11=264 input features (25-1 for the center cell which is the one we are predicting). This can be any length really. 

After initialization you need to set up the weights and biases with their appropriate values. This takes in the weights and biases as an array and assigns them to the network.

Finally you call `evaluate` with the inputs (i.e. the cell values if using the encoding scheme above), and it returns the output value (prediction) of the neural network.

Keep in mind this is highly optimized for what I want to be doing with my PSO so some of the design decisions might not make complete sense.

You can have a look at the `runner.py` to see it in action.
# 700-copy-2
