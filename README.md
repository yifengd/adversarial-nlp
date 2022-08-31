# Adversarial Text in NLP: Improving robustness via XAI

Code for the paper "Improving Adversarial Robustness against Sentence-level Text Attacks using Explainable AI".

**Authors:** Yifeng Dong, Niklas Hölterhoff, Karl Richter

## Code usage guide

### Prerequisites

This codebase is designed for use with Google Colab and Google Drive. Each `.ipynb` notebook should be opened in Google Colab.

The codebase reads from and writes to your Google Drive, to a folder named `AdversarialXAI` in the root directory. Before running the code, unzip and upload the `data/AdversarialXAI.zip` folder to your Google Drive. If you do this, you can start running the defense methods immediately.

### Running the WDR-based methods

To train and test the WDR-based methods, use the `defenses/wdr/wdr.ipynb` notebook.

### Running the Layer-attribution methods

To train and test the Layer-attribution method, use the notebooks under `defenses/layer-attribution/`. The `base.ipynb` contains the default code to train the base adversarial detector. The `classifiers.ipynb` contains additional classifiers we used for our experiments. The `visual_inspection.ipynb` contains code to generate graphics that allow for the visual inspection of the layer-attributions.

### Data generation

To generate your own StyleAdv adversarial samples, use the `attacks/styleadv/StyleAdv.ipynb` notebook.

- If you want to use more data from the [Yoo et. al. benchmark](https://github.com/bangawayoo/adversarial-examples-in-text-classification), use the `attacks/Benchmark_Datasets.ipynb` notebook to preprocess the data.

- To download more corpuses from Huggingface, use the `attacks/Huggingface_Datasets.ipynb` notebook.

## Acknowledgments

This work builds heavily on the [WDR](https://github.com/javirandor/wdr) repository by Mosca et al.

Note: The `StyleAttack.zip` archive is a copy of https://github.com/yifengd/StyleAttack, which is a fork of https://github.com/thunlp/StyleAttack. The copy is included for submission purposes only.
