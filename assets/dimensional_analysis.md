## Dimensional Impact Analysis

| model       | dimension   | value1     | value2     |   acc_mean_1 |   acc_mean_2 |   acc_std_1 |   acc_std_2 |   acc_p_value | acc_significant   |   group_cons_1 |   group_cons_2 |
|:------------|:------------|:-----------|:-----------|-------------:|-------------:|------------:|------------:|--------------:|:------------------|---------------:|---------------:|
| gpt-4.1     | formality   | casual     | formal     |       0.9551 |       0.9561 |      0.0018 |      0.0014 |        0.2241 |                   |         0.9075 |         0.9225 |
| gpt-4o-mini | formality   | casual     | formal     |       0.9496 |       0.952  |      0.0068 |      0.0032 |        0.3785 |                   |         0.9075 |         0.9075 |
| gpt-4.1     | phrasing    | imperative | question   |       0.9553 |       0.9558 |      0.0011 |      0.0022 |        0.5809 |                   |         0.9125 |         0.9175 |
| gpt-4o-mini | phrasing    | imperative | question   |       0.9507 |       0.9509 |      0.0052 |      0.0057 |        0.9399 |                   |         0.9125 |         0.8925 |
| gpt-4.1     | order       | task_first | text_first |       0.9562 |       0.955  |      0.0012 |      0.0019 |        0.1682 |                   |         0.9525 |         0.9225 |
| gpt-4o-mini | order       | task_first | text_first |       0.9529 |       0.9487 |      0.0033 |      0.0061 |        0.1146 |                   |         0.945  |         0.945  |
| gpt-4.1     | synonyms    | set_a      | set_b      |       0.9557 |       0.9555 |      0.0012 |      0.0021 |        0.8131 |                   |         0.9075 |         0.9375 |
| gpt-4o-mini | synonyms    | set_a      | set_b      |       0.9542 |       0.9474 |      0.0028 |      0.005  |        0.0068 | ✓                 |         0.905  |         0.925  |