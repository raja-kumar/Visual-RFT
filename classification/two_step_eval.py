import json

def calculate_accuracies(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    total = 0
    topk_correct = 0
    step2_correct = 0
    consistency_correct = 0

    for k, v in data.items():
        # skip non-image keys
        if not isinstance(v, dict):
            continue
        gt = v.get('groundtruth')
        step1_output = v.get('step1_output', {})
        prediction = v.get('prediction')

        if gt is None or not step1_output:
            continue

        total += 1

        # 1. topk accuracy: groundtruth in step1_output keys
        if gt in step1_output:
            topk_correct += 1

        # 2. step2 accuracy: prediction == groundtruth
        if prediction == gt:
            step2_correct += 1
        else:
            print(f"Mismatch: {k}, GT: {gt}, Prediction: {prediction}")
        
        top_prediction = sorted(step1_output.items(), key=lambda x: x[1], reverse=True)[0][0] if step1_output else None
        if top_prediction == gt:
            consistency_correct += 1

    topk_acc = topk_correct / total if total else 0
    step2_acc = step2_correct / total if total else 0
    consistency_acc = consistency_correct / total if total else 0

    print(f"Total samples: {total}")
    print(f"TopK accuracy: {topk_acc:.4f}")
    print(f"Step2 accuracy: {step2_acc:.4f}")
    print(f"Consistency accuracy: {consistency_acc:.4f}")

if __name__ == "__main__":
    calculate_accuracies(
        "/app/Visual-RFT/classification/output/oxford_flowers/two_steps/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_qwen_mcq_checkpoint-400_base_val.json"
    )