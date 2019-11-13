import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

arr = ["affenpinscher","afghan_hound","african_hunting_dog","airedale","american_staffordshire_terrier","appenzeller","australian_terrier","basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog","black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick","border_collie","border_terrier","borzoi","boston_bull","bouvier_des_flandres","boxer","brabancon_griffon","briard","brittany_spaniel","bull_mastiff","cairn","cardigan","chesapeake_bay_retriever","chihuahua","chow","clumber","cocker_spaniel","collie","curly-coated_retriever","dandie_dinmont","dhole","dingo","doberman","english_foxhound","english_setter","english_springer","entlebucher","eskimo_dog","flat-coated_retriever","french_bulldog","german_shepherd","german_short-haired_pointer","giant_schnauzer","golden_retriever","gordon_setter","great_dane","great_pyrenees","greater_swiss_mountain_dog","groenendael","ibizan_hound","irish_setter","irish_terrier","irish_water_spaniel","irish_wolfhound","italian_greyhound","japanese_spaniel","keeshond","kelpie","kerry_blue_terrier","komondor","kuvasz","labrador_retriever","lakeland_terrier","leonberg","lhasa","malamute","malinois","maltese_dog","mexican_hairless","miniature_pinscher","miniature_poodle","miniature_schnauzer","newfoundland","norfolk_terrier","norwegian_elkhound","norwich_terrier","old_english_sheepdog","otterhound","papillon","pekinese","pembroke","pomeranian","pug","redbone","rhodesian_ridgeback","rottweiler","saint_bernard","saluki","samoyed","schipperke","scotch_terrier","scottish_deerhound","sealyham_terrier","shetland_sheepdog","shih-tzu","siberian_husky","silky_terrier","soft-coated_wheaten_terrier","staffordshire_bullterrier","standard_poodle","standard_schnauzer","sussex_spaniel","tibetan_mastiff","tibetan_terrier","toy_poodle","toy_terrier","vizsla","walker_hound","weimaraner","welsh_springer_spaniel","west_highland_white_terrier","whippet","wire-haired_fox_terrier","yorkshire_terrier"]

class Dataset(Dataset):

	def __init__(self):
		self.csv = pd.read_csv('./labels.csv')

	def __len__(self):
		return len(self.csv)

	def __getitem__(self, ind):
		name = self.csv.iloc[ind][0] + ".jpg"
		X = cv2.imread("./train_resized/" + name)
		X = X.reshape(3, 224, 224)
		# print(X.shape)
		# print(type(X))
		# print(len(X), len(X[0]), len(X[0][0]))
		# y_arr = np.zeros(120)
		# y_arr[arr.index(self.csv.iloc[ind][1])] = 1
		Y = arr.index(self.csv.iloc[ind][1])
		# print(Y)
		
		X = torch.from_numpy(X).float()

		return X, Y

