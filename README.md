Two approaches for this problem

1. Label the images as odd even (1/0) and model a binary classification problem
2. Model a multiclass classification problem, and apply odd/even filter on top of its predictions

The latter gives accuracy improvement of 96% over the former; while the former fails to perform good on binary setting

```bash
python3 -m venv mnist_tf_env
source mnist_tf_env/bin/activate

# Train the model
python train.py

# to test sample images
python test.py
```