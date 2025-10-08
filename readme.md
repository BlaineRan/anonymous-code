# Setup and Run Guide

This document provides instructions on how to install dependencies, configure the project, and run it successfully.

---

## 1. Install Dependencies

Both **Conda** and **Pip** dependencies must be installed for the project to run properly.  
Please execute **both** of the following commands in order:

### Step 1: Install Conda Dependencies

```bash
conda install --file conda_requirements.txt
```
### Step 2: Install Conda Dependencies
```bash
conda install --file conda_requirements.txt
```

## 2. Configuration
Before running the project, edit the configuration file located at:

```bash
configs/llm_config.yaml
```

Update the following fields with your actual information:

```yaml
llm:
  model_name: ""          # Specify the model name
  temperature:            # Set a suitable temperature value (e.g., 0.7)
  base_url: ""            # Replace with your actual API base URL
  api_key: ""             # Replace with your actual API key
```

## 3. Run
After installing all dependencies and updating the configuration, run with:

```bash
python /root/tinyml/mcts/mcts_searcher.py --max_peak_memory 15 --dataset_name Mhealth
```