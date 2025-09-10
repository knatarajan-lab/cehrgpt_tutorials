# CEHR-GPT Synthetic Data Generation Tutorials

This repository provides comprehensive tutorials demonstrating the capabilities of CEHR-GPT for generating high-quality synthetic healthcare data. These step-by-step guides will walk you through the complete pipeline from setup to synthetic data generation.

You can find a working example in a Colab notebook at https://colab.research.google.com/drive/19Fli5zSe1a3HQ-EQA6v219GrhixdaY0p

## Prerequisites

### Environment Setup

First, establish a clean Python environment to ensure compatibility and avoid dependency conflicts:

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install cehrgpt --constraint constraints.txt
pip install gdown
```

Next, create the necessary directory structure for organizing your data and models:

```shell
mkdir omop_synthea
mkdir omop_synthea/cehrgpt
mkdir omop_synthea/dataset_prepared
mkdir omop_synthea/cehrgpt/synthetic_data
```

Configure the environment variables that will be used throughout the pipeline. These variables define the paths for your OMOP data, CEHR-GPT models, and output directories:

```shell
export OMOP_DIR=omop_synthea
export CEHR_GPT_DATA_DIR=omop_synthea
export CEHR_GPT_MODEL_DIR=omop_synthea/cehrgpt
export SYNTHETIC_DATA_OUTPUT_DIR=omop_synthea/cehrgpt/synthetic_data
```

### PySpark Environment Configuration

CEHR-GPT leverages Apache Spark for efficient large-scale data processing. Configure your Spark environment with the following settings:

```shell
# Set Spark home directory
export SPARK_HOME=$(python -c "import pyspark; print(pyspark.__file__.rsplit('/', 1)[0])")

# Configure Python interpreters for Spark processes
export PYSPARK_PYTHON=$(python -c "import sys; print(sys.executable)")
export PYSPARK_DRIVER_PYTHON=$(python -c "import sys; print(sys.executable)")

# Update Python and system paths
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$PATH

# Configure Spark resource allocation
export SPARK_WORKER_INSTANCES=1
export SPARK_WORKER_CORES=16
export SPARK_EXECUTOR_CORES=8
export SPARK_DRIVER_MEMORY=20g
export SPARK_EXECUTOR_MEMORY=20g
export SPARK_MASTER=local[16]

export SPARK_SUBMIT_OPTIONS="--master $SPARK_MASTER --driver-memory $SPARK_DRIVER_MEMORY --executor-memory $SPARK_EXECUTOR_MEMORY --executor-cores $SPARK_EXECUTOR_CORES"
```

### Download Sample Data

We provide a pre-generated Synthea dataset containing 1 million synthetic patients to help you get started quickly. This dataset follows the OMOP Common Data Model format and provides a realistic foundation for testing CEHR-GPT:

```shell
gdown --fuzzy "https://drive.google.com/file/d/1k7-cZACaDNw8A1JRI37mfMAhEErxKaQJ/view?usp=share_link"
```

Extract the downloaded archive to your designated OMOP directory:

```shell
tar -xaf omop_synthea.tar.gz -C omop_synthea
```

## Step 1: Generate Training Data

Transform your OMOP-formatted healthcare data into sequences suitable for training CEHR-GPT. This preprocessing step converts patient records into tokenized sequences that capture temporal relationships and clinical events:

```shell
sh scripts/create_cehrgpt_pretraining_data.sh \
    --input_folder $OMOP_DIR \
    --output_folder $CEHR_GPT_DATA_DIR \
    --start_date 1985-01-01
```

This script processes the raw OMOP data and creates structured patient sequences, handling temporal ordering, concept mapping, and data formatting required for the transformer architecture.

## Step 2: Train CEHR-GPT

Train your CEHR-GPT model using the preprocessed patient sequences. This transformer-based model learns to generate realistic healthcare trajectories by understanding patterns in patient data:

```shell
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
    --model_name_or_path $CEHR_GPT_MODEL_DIR \
    --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
    --output_dir $CEHR_GPT_MODEL_DIR \
    --data_folder $CEHR_GPT_DATA_DIR/patient_sequence/train \
    --dataset_prepared_path $CEHR_GPT_DATA_DIR/dataset_prepared \
    --do_train true \
    --seed 42 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 8 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --max_position_embeddings 1024 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --sample_packing \
    --max_tokens_per_batch 8192 \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --num_train_epochs 10 \
    --learning_rate 0.0001 \
    --use_early_stopping \
    --load_best_model_at_end true \
    --early_stopping_threshold 0.001
```

The training process uses a 12-layer transformer architecture with 768 hidden dimensions, optimized for healthcare sequence modeling. Training includes early stopping to prevent overfitting and ensure optimal model performance.

## Step 3: Generate Synthetic Sequences

Use your trained CEHR-GPT model to generate new synthetic patient sequences. The generation process employs sophisticated sampling strategies to produce diverse and realistic healthcare trajectories:

```shell
export TRANSFORMERS_VERBOSITY=info
export CUDA_VISIBLE_DEVICES="0"
python -u -m cehrgpt.generation.generate_batch_hf_gpt_sequence \
    --model_folder $CEHR_GPT_MODEL_DIR \
    --tokenizer_folder $CEHR_GPT_MODEL_DIR \
    --output_folder $SYNTHETIC_DATA_OUTPUT_DIR \
    --num_of_patients 128 \
    --batch_size 16 \
    --buffer_size 128 \
    --context_window 1024 \
    --sampling_strategy TopPStrategy \
    --top_p 1.0 \
    --temperature 1.0 \
    --repetition_penalty 1.0 \
    --epsilon_cutoff 0.00 \
    --demographic_data_path $CEHR_GPT_DATA_DIR/patient_sequence/train
```

This command generates 128 synthetic patients using top-p sampling with configurable parameters for controlling randomness and diversity in the generated sequences. The model considers a context window of 1024 tokens to maintain long-range dependencies in patient trajectories.

## Step 4: Convert Synthetic Sequences to OMOP

Transform the generated synthetic sequences back into standard OMOP Common Data Model format. This final step ensures that your synthetic data maintains the same structure and relationships as real healthcare data:

```shell
sh scripts/omop_pipeline.sh \
    --patient-sequence-folder=$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/ \
    --omop-folder=$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/restored_omop/ \
    --source-omop-folder=$OMOP_DIR \
    --cpu-cores=10
```

The conversion process reconstructs proper OMOP tables (Person, Condition_Occurrence, Drug_Exposure, etc.) from the synthetic sequences, ensuring data integrity and compatibility with existing healthcare analytics tools and workflows.

Upon completion, you'll have a complete set of synthetic healthcare data in OMOP format that preserves the statistical properties and clinical patterns of the original dataset while protecting patient privacy.