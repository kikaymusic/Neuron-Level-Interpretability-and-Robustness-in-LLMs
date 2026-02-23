# LUMIA

This guide provides instructions to set up the environment and install the necessary dependencies for the project.

---

## Prerequisites

Before starting, ensure the following tools are installed on your system:

- **Python 3.10.14**

- **Anaconda** or **Miniconda** (optional)

---

## Setup Instructions

### 1. Clone the Repository

To begin, clone the repository to your local machine and navigate to the project directory:

```bash
git clone <repo-name-here>
cd lumia
```

---

### 2. Create a New Environment (Optional)

Create and activate a new Conda environment for the project:

```bash
conda create -n lumia python=3.10.14 -y
conda activate lumia
```

---

### 3. Install Dependencies

Install all necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

### 4. Use the Automated Script (Optional)

For convenience, you can use the provided automated setup script. Ensure it has executable permissions before running:

```bash
cd scripts
sudo chmod +x setup_environment.sh
./setup_environment.sh
```

---

## Running Experiments

### Unimodal Experiments

#### For NGB:
```bash
cd scripts
sudo chmod +x run_unimodal_ngb_experiment.sh
./run_unimodal_ngb_experiment.sh
```

#### For TB:
```bash
cd scripts
sudo chmod +x run_unimodal_tb_experiment.sh
./run_unimodal_tb_experiment.sh
```

### Multimodal Experiments

Run the multimodal experiments script:

```bash
cd scripts
sudo chmod +x run_multimodal_experiment.sh
./run_multimodal_experiment.sh
```

---

## About the Data

### Datasets

- **NGB and Multimodal Experiments**: Most datasets are sourced from Hugging Face. 
- **TB Experiments**: Navigate to the `./data` directory and execute the download script for the Pajama dataset used in the `arxiv-1-month` experiment.

### Additional Data Sources

- For **PG-19 (Gutenberg Experiment)**:  
  Download the data from [DeepMind Gutenberg](https://console.cloud.google.com/storage/browser/deepmind-gutenberg) and place it in the `./data/pg19` folder.

---
