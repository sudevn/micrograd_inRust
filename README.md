# Micrograd - Rust Implementation

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/sudevn/micrograd_inRust/build.yml)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/sudevn/micrograd_inRust)

This project is a Rust implementation of the introductory neural network and backpropagation example presented in the article ["The spelled-out intro to neural networks and backpropagation: building micrograd" by Andrej Karpathy](https://youtu.be/VMj-3S1tku0?si=V_zYU6_ddAUqC7ld).

## Introduction

Micrograd is a minimal autograd engine that performs automatic differentiation of computational graphs. This Rust implementation aims to provide a clear and concise implementation of the concepts discussed in the article, making it accessible to Rust developers interested in deep learning and neural networks.

## Features

- Implements a simple neural network with one hidden layer.
- Demonstrates forward propagation, loss calculation, and backpropagation.
- Utilizes gradient descent for optimization.
- Designed to be easy to understand and modify.

## Getting Started

To get started with Micrograd, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/sudevn/micrograd_inRust.git
   ```

2. Navigate to the project directory:

   ```bash
   cd micrograd-rust
   ```

3. Build and run the example:

   ```bash
   cargo run --example neural_network
   ```

## Usage

You can use Micrograd in your own projects by adding it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
micrograd = { git = "https://github.com/sudevn/micrograd_inRust.git" }
```

Then, import Micrograd into your Rust code:

```rust
extern crate micrograd;
use micrograd::*;
```

Now you can use Micrograd's functionality in your project.

## Contributing

Contributions to Micrograd are welcome! If you find any bugs or have ideas for improvements, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Andrej Karpathy for the original article and inspiration.
- Contributors to the Micrograd project.
- The Rust community for creating an amazing language.

---
