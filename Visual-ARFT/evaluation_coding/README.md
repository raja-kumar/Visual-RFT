# Evaluation on MAT-Coding
In this folder, we provide two Python files for evaluation on **MAT-Coding**.
``evaluation_mat_coding_ngpu_7b_df.py`` is used to evaluate the results of direct inference, while `evaluation_mat_coding_visual_arft.py` is used to evaluate the inference results of the **Visual-Agentic-Coding** model.

## Download Dataset and Models
You can download MAT Benchmakr on 🤗<a href="https://huggingface.co/datasets/laolao77/MAT">Datasets</a>. Use `/MAT/MAT-Benchmark/MAT-Coding.json`.

You can download our model: 🤗<a href="https://huggingface.co/laolao77/Visual-ARFT-Coding">Visual-ARFT-Coding</a></h3>

## Evaluation on MAT-Coding

To run `evaluation_mat_coding_visual_arft.py`, you need to replace the paths to the model and dataset:

-  Line 175: Replace **model_name** with the actual model path (Visual-ARFT-Coding).
-  Line 184: Replace **json_path** with the actual dataset path (MAT-Coding.json).
-  Line 201: Replace **input_image_path** with the actual image path.
-  Line 342: Set the results save path.

> 🔔The intermediate image processing result will be saved as `cache.jpg` in the current directory, as specified in Line 206.

To run `evaluation_mat_coding_ngpu_7b_df.py`, you need to replace the paths to the model and dataset:

-  Line 20: Replace **model_name** with the actual model path (Visual-ARFT-Coding).
-  Line 30: Replace **json_path** with the actual dataset path (MAT-Coding.json).
-  Line 42: Replace **input_image_path** with the actual image path.
-  Line 103: Set the results save path.







