#!/bin/bash
# GLM-4-9B 心理学专项评测脚本
# 评测多个心理学相关的测试集

evalscope eval \
  --model glm4-9b-sft \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets mmlu mmlu_pro mmlu_redux super_gpqa \
  --dataset-args '{
    "mmlu": {
      "subset_list": ["professional_psychology", "high_school_psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "mmlu_pro": {
      "subset_list": ["psychology"],
      "few_shot_num": 5,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "mmlu_redux": {
      "subset_list": ["professional_psychology", "high_school_psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "super_gpqa": {
      "subset_list": ["Psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    }
  }'
