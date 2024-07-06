# Time and space test

| device | all on gpu | divide into two blocks | device in four blocks |
|--------|------------|------------------------|-----------------------|
| A100   | 9s, 13G    | 28-31s, 8.8-9GB        | 44s, 4-5GB            |   
| 4090   | 18s, 13GB  | 59-63s, 8.8-9.4GB      | 1m20s, 4-5G           |