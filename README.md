# Selective Retrieval-Augmentation for Long-Tail Legal Text Classification

This repository contains the code for the paper:

**Selective Retrieval-Augmentation for Long-Tail Legal Text Classification**  
_Boheng Mao_ 

## Overview

Proposed a **Selective Retrieval-Augmentation (SRA)** strategy that selectively augments low-frequency class examples using retrieved clauses to improve classification on long-tailed legal datasets.

Tested on:
- **LEDGAR** (multi-class)
- **UNFAIR-ToS** (multi-label)

SRA improves performance especially on rare classes.

## Project Structure

The project is organized by dataset, with separate folders for **LEDGAR** and **UNFAIR-ToS**. Each dataset folder contains everything needed to run experiments — from data augmentation to training and analysis. Here's a quick breakdown of what each part does:

- `ledgar/`  
  All code related to the LEDGAR dataset.
  - `generate_sra_dataset/`: Code for generating selectively augmented training data using retrieval-based augmentation (SRA). This includes retrieving similar clauses with TF-IDF and SBERT.
  - `train/`: Training scripts for models **with** and **without** augmentation. These scripts use the same architecture and differ only in whether the input has been augmented.
  - `analysis/`: Scripts for post-hoc analysis — checking performance improvements, visualizing results, measuring coverage, similarity, and running case studies.

- `unfair_tos/`  
  Follows the same structure as LEDGAR, but adapted to the UNFAIR-ToS dataset (which is a multi-label classification task).
  - Includes both `sra.py` (selective) and `full_retrieval.py` (non-selective) augmentation methods for comparison.
  - Training and analysis scripts are similar, just adapted for the label format and task differences.
