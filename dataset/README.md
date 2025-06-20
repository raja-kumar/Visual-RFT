#### steps to generate the MCQ dataset

## generate the json file for virft dataset
python test_dataset.py

## using the above generated json file, generate the top5 by using llm. for this run,

python /home/raja/OVOD/git_files/VLM-COT/evaluate.py # change the config and make sure to use evaluate_v2

## finally run the steps in /home/raja/OVOD/git_files/Visual-RFT/dataset/generate_mcq.ipynb

## run /home/raja/OVOD/git_files/Visual-RFT/dataset/build_dataset.ipynb to convert it to huggingface format