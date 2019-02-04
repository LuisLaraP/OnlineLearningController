# Online Learning Controller

A robot controller that learns on-line and adapts to changing conditions.

## Installation

The core functionality (enough in most situations) is installed by executing the following command in the root directory of the project:

```bash
pip install .
```

### Extra functionality

It is possible to run experiments on the environments provided by OpenAI Gym. To do this, install the package with the following command instead:

```bash
pip install '.[openai]'
```

## Usage

This project provides a single entry point, `olc`. This program reads an specification file in JSON format and runs an experiment defined by the specifications given.

The command line interface for the `olc` program is:

```bash
olc <filename>
```

where *filename* should be replaced with the path to the desired specification file.

For a more detailed summary of the command line options for the `olc` program, execute:

```bash
olc -h
```

## Author

Luis Alejandro Lara Patiño ([luislpatino@gmail.com](mailto:luislpatino@gmail.com))
