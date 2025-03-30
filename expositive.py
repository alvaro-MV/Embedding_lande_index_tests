import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from getpass import getpass
import embDataset
import embUtils as utils
from embRun import run_on_expositive_texts

expositive = embDataset.KaggleDataset(
	"nechbamohammed/research-papers-dataset")

expositive.load_dataset()
expositive.transform_dataset()