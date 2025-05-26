## Best Performing Combinations

| model   | variant_id   | formality   | phrasing   | order      | synonyms   |   custom_accuracy |   model_consistency |   weighted_index |
|:--------|:-------------|:------------|:-----------|:-----------|:-----------|------------------:|--------------------:|-----------------:|
| gpt-4.1 | v6           | formal      | question   | task_first | set_b      |             0.958 |               0.912 |            0.944 |
| gpt-4.1 | v8           | formal      | question   | text_first | set_b      |             0.958 |               0.912 |            0.944 |
| gpt-4.1 | v1           | formal      | imperative | task_first | set_a      |             0.957 |               0.912 |            0.944 |
| gpt-4.1 | v13          | casual      | question   | task_first | set_a      |             0.957 |               0.912 |            0.944 |
| gpt-4.1 | v7           | formal      | question   | text_first | set_a      |             0.956 |               0.912 |            0.943 |

## Worst Performing Combinations

| model       | variant_id   | formality   | phrasing   | order      | synonyms   |   custom_accuracy |   model_consistency |   weighted_index |
|:------------|:-------------|:------------|:-----------|:-----------|:-----------|------------------:|--------------------:|-----------------:|
| gpt-4o-mini | v12          | casual      | imperative | text_first | set_b      |             0.94  |               0.902 |            0.929 |
| gpt-4o-mini | v16          | casual      | question   | text_first | set_b      |             0.941 |               0.902 |            0.929 |
| gpt-4o-mini | v14          | casual      | question   | task_first | set_b      |             0.947 |               0.902 |            0.934 |
| gpt-4o-mini | v4           | formal      | imperative | text_first | set_b      |             0.948 |               0.902 |            0.934 |
| gpt-4o-mini | v8           | formal      | question   | text_first | set_b      |             0.948 |               0.902 |            0.935 |