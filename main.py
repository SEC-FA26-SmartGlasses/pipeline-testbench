from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline

cases = load_dataset("E:/test_dataset")
# I'm not putting the dataset in the repo because it's like 100 megabytes
# we need to decide another place to host it 
print(cases)


