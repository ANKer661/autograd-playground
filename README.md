# Autograd Playground

Autograd Playground is a simple implementation of automatic differentiation in Python & Numpy. This project allows users to create computation graphs, visualize them, and perform back propagation to understand how gradients flow between tensors.

## Features

- Simple `Tensor` class for creating and manipulating tensors
- Support for basic mathematical operations: (element-wised `+`, `-`, `*`, `/`) and matrix multiplication
- Automatic computation graph construction
- Back-propagation for gradient computation
- Visualization of computation graphs before and after back-propagation

## Project Structure
- `src`
  - `tensor.py`: Contains the `Tensor` class implementation
  - `operations.py`: Defines various mathematical operations (Add, Subtract, Multiply, etc.)
  - `visualization.py`: Provides functions for visualizing computation graphs
- `autograd-playground.ipynb`: Jupyter notebook with examples and explanations


## Installation

1. Clone this repository:
```bash
git clone https://github.com/ANKer661/autograd-playground.git
cd autograd-playground
```

2. Install required dependencies:
```bash
pip install numpy matplotlib networkx
```

Or try out **Binder** for a quick start without any local installation. Click the badge below to launch the project in a Binder environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ANKer661/autograd-playground/main?labpath=autograd-playground.ipynb)

## Usage

The main interface for this project is through the Jupyter notebook `autograd-playground.ipynb`. To run the notebook:

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `autograd-playground.ipynb` in your browser.

3. Run the cells in the notebook to create tensors, build computation graphs, and visualize the results.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
