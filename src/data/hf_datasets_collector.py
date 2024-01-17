import datasets
from src.data.preprocessing import PreprocessingUtils

# Load the lite configuration of the dataset
raw_dataset = datasets.load_dataset("JanosAudran/financial-reports-sec", "large_lite")
print(raw_dataset)

