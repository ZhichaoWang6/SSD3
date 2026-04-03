export LLM_EVALUATOR_URL="https://your_api_url/v1/openai/native/chat/completions"
export LLM_EVALUATOR_MODEL="gpt-4.1"
llm_evaluator="gpt-4.1"
exp_name=mmduet2
num_workers=10

dataset=ego
output_dir=outputs/${exp_name}/${dataset}

cd "$(dirname "$0")/.."

python -u evaluate.py --num_workers ${num_workers} \
    --gold_file ./data/annotations/${dataset}-proactivevideoqa_format.json \
    --input_file $output_folder/pred.jsonl --output_file $output_folder/${llm_evaluator}-eval.json \
    > $output_folder/${llm_evaluator}-eval.log 2>&1 &