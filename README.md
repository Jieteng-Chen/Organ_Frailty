# Frailty Index Prompt for LLM-based Prediction

This repository contains the custom prompt used in **Organ-specific frailty transitions predict mortality during human aging** for LLM-based estimation of the overall and system-specific Frailty Index (FI).

## Relationship to the original code

Our inference pipeline is built on the code released with:

> *Large language model-based biological age prediction in large-scale populations.* Nature Medicine, 2025. https://doi.org/10.1038/s41591-025-03856-8
> Code: https://github.com/AI4HEALTH-LAB-THU/LLM-Aging

We replaced the prompt file `vllm_inference/prompts/main_analysis.txt` with the frailty-specific prompt provided here (`prompt_frailty_index.txt`).

## Contents

| File | Description |
|---|---|
| `prompt_frailty_index.txt` | Prompt for estimating overall FI and nine system-specific FIs from individual-level health information. |


## Output

The model returns a JSON object with a reasoning field and a value (float, 0 to 1) for each of the following:

overall, cardiovascular, hepatic, gastrointestinal, urinary, immune, endocrine, musculoskeletal, neural, and respiratory FI.

## How to reproduce

1. Clone the original repository:
   ```bash
   git clone https://github.com/AI4HEALTH-LAB-THU/LLM-Aging.git
   cd LLM-Aging
   git checkout
   ```
2. Copy `prompt_frailty_index.txt` from this repository into `vllm_inference/prompts/`, replacing `main_analysis.txt` (either keep the file name `main_analysis.txt`, or update the prompt path in the inference script).
3. Prepare the input data in the same format as the original pipeline and run the vLLM inference following the original README.
4. Parse the JSON output to obtain the FI estimates.
