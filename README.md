# Hierarchical Multi-Agent DRL for Dynamic Flexible Job-Shop Scheduling with Transportation (DFJSP-T)

## Overview

This project provides a solution for the **Dynamic Flexible Job-Shop Scheduling Problem with Transportation** (DFJSP-T) using a **Hierarchical Multi-Agent Deep Reinforcement Learning (DRL)** framework combined with **Imitation Learning (IL)**. The approach aims to optimize job-shop scheduling by integrating the transportation of robots (transbots) for job completion. It decomposes the problem into three decision levels:

1. **High-level Agent:** Prioritizes jobs to be processed.
2. **Mid-level Agent:** Selects machines for each assigned job.
3. **Low-level Agent:** Manages the transportation (transbot) between machines and jobs.

The project includes several components such as environment setups, agent models, dispatching rules, data generation, and training scripts, which together help to solve the scheduling problem efficiently.

## Project Structure

### `DFJSPT` (Dynamic Flexible Job-Shop Scheduling with Transportation)
- **`dfjspt_data/`**: Contains utilities and data for generating and loading scheduling instances.
    - `dfjspt_data_generator.py`: Generates synthetic scheduling data for testing and training.
    - `Hurink_data/`: Stores data from a publicly available dataset for the flexible job-shop problem.
    - `load_data_from_Ham.py `: Loads data from the "Hurink_data" dataset.
    - `generate_ham_data.py`: Creates synthetic data using the same distribution with "Hurink_data" dataset to simulate scheduling instances.
    - `radar_microwave_case_data.py`: Generates scheduling data for a specific real-world scenario.

- **`dfjspt_rule/`**: Contains multiple dispatching rule implementations for job and machine selection.
    - Each `dfjspt_ruleX.py` file defines a specific dispatching rule for job scheduling (e.g., based on earliest start time, shortest processing time).
    - `dfjspt_rule_compare.py`: A script to compare the performance of various dispatching rules.
    - `dfjspt_rule_test_benchmark.py`: A script for benchmarking different dispatching rules on "Hurink_data" dataset.
    - `job_selection_rules.py` and `machine_selection_rules.py`: Define multiple strategies for selecting jobs and machines in the scheduling process.

- **`dfjspt_params.py`**: Configuration file that defines various parameters needed for solving the scheduling problem.

- **`dfjspt_env.py`**: The main environment setup for the DRL framework, where agents interact with the scheduling environment.

- **`env_for_rule.py`**: A simplified version of the DRL environment that integrates dispatching rules instead of DRL-based agents.

- **`dfjspt_env_for_benchmark.py`**: A specialized environment setup for benchmarking scheduling solutions on "Hurink_data" dataset.

- **`dfjspt_agent_model.py`**: Defines the models of the reinforcement learning agents used in this project, including the high, mid, and low-level agents.

- **`dfjspt_train.py`**: The script used to train the DRL agents, including setting up the environment and training loops.

- **`dfjspt_train_case.py`**: A script to train the agents on the specific, real-world case defined in `radar_microwave_case_data.py`.

- **`dfjspt_test.py`**: A testing script that evaluates the performance of the trained agents on scheduling tasks.


## Installation

Clone this repository to your local machine:
   ```bash
   git clone https://github.com/ClouDaDaDa/dfjspt_code.git
   cd dfjsp_code
   ```

Ensure that you have all the required dependencies installed by running:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model
To train the model for DFJSP-T, use the following script:
```bash
python DFJSPT/dfjspt_train.py
```

This will initialize the environment, set up the agent, and start the training loop. The training process will utilize both DRL and IL strategies to optimize scheduling.

By default, the training script is configured to run with `num_workers=4` to ensure compatibility with small computers. If you'd like to use multiple CPUs for parallel environment sampling, you can increase this parameter in the script.

During training, the script will automatically create subdirectories to save the training results and logs. The directory structure will be organized as follows:

```php-template
  training_results/
    ├── J<job_count>_M<machine_count>_T<transbot_count>
```

Where:
- `J<job_count>` refers to the number of jobs.
- `M<machine_count>` refers to the number of machines.
- `T<transbot_count>` refers to the number of transbots.

### Testing a Model
To test a trained model:
```bash
python DFJSPT/dfjspt_test.py
```

This will evaluate the trained agent on specified test environments and provide performance metrics for the scheduling solution.

### Running Experiments
To run experiments using different dispatching rules, you can execute the scripts located in the `DFJSPT/dfjspt_rule/` directory. These scripts contain various predefined dispatching rules for job selection, machine selection, and transbot scheduling. 

For example:

```bash
  python DFJSPT/dfjspt_rule/dfjspt_rule_compare.py
```

## Configuration

All environment configurations and training configurations can be modified in the `dfjspt_params.py` file. 

