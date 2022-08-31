# Adversarial Text in NLP: Improving robustness via XAI

Code for the paper "Improving Adversarial Robustness against Sentence-level Text Attacks using Explainable AI".

Authors: Yifeng Dong, Niklas Hölterhoff, Karl Richter

## Code usage guide

### Prerequisites

- This codebase is designed for use with Google Colab and Google Drive. Each `.ipynb` notebook should be opened in Google Colab.

- The codebase reads from and writes to your Google Drive, to a folder named `AdversarialXAI`. Before running the code, unzip and upload the `data/AdversarialXAI.zip` folder to your Google Drive. If you do this, you can start running the defense methods immediately.

### Running the WDR-based methods

- To train and test the WDR-based methods, use the `defenses/wdr/wdr.ipynb` notebook.

### Running the Captum-based methods

tbd

### Data generation

- To generate your own StyleAdv adversarial samples, use the `attacks/styleadv/StyleAdv.ipynb` notebook.

- If you want to use more data from the [Yoo et. al. benchmark](), use the `attacks/Benchmark_Datasets.ipynb` notebook to preprocess the data.

- To download more corpuses from Huggingface, use the `attacks/Huggingface_Datasets.ipynb` notebook.

## Acknowledgments

This work builds heavily on the [WDR](https://github.com/javirandor/wdr) repository by Mosca et al.

