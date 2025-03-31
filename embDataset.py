from abc import ABC, abstractmethod
import os
import pandas as pd
import random
import kagglehub
from datasets import load_dataset


class embDataset(ABC):
	@abstractmethod	
	def load_dataset(self):
		pass

	@abstractmethod
	def transform_dataset(self, transform_func):
		pass

	@abstractmethod
	def get_df_from_dataset(self):
		pass
	
	@abstractmethod
	def get_df(self):
		pass


class HFDataset(embDataset):
	def __init__(self, path):
		self.path = path
		self.df = None
		self.hf_dataset = None

	def load_dataset(self):
		self.hf_dataset = load_dataset(self.path)

	def transform_dataset(self, transform_func):
		self.hf_dataset = transform_func(self.hf_dataset)

	def get_df_from_dataset(self):
		self.df = pd.DataFrame(self.hf_dataset)

	def transform_df(self, transform_func):
		self.df = transform_func(self.df)
	
	def	get_df(self):
		return (self.df)

	def get_sample(self, k = 5):
		return self.df.iloc[random.choices(
			range(0, self.df.index.stop -1), k = k)]


class KaggleDataset(embDataset):
	def __init__(self, path):
		self.path = path
		self.dataset = None
		self.df = None

	def load_dataset(self):
		self.dataset = kagglehub.dataset_download(self.path)
		print("Path to dataset files:", self.dataset)
	
	def transform_dataset(self, transform_func):
		self.dataset = transform_func(self.dataset)

	def get_df_from_dataset(self):
		csv_file=os.listdir(self.dataset)
		csv_file_path = os.path.join(self.dataset, csv_file[0])
		self.df = pd.read_csv(csv_file_path)
		print(self.df.head(10))
	
	def	get_df(self):
		return (self.df)

	def get_sample(self, k = 5):
		return self.df.iloc[random.choices(
			range(0, self.df.index.stop -1), k = k)]

