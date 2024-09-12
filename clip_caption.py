import os
import time
import re
import json
from clip_retrieval.clip_client import ClipClient, Modality
from langdetect import detect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

img_folder_path = "/datasets/images/query"
caption_path = "/datasets/download/caption"
class_path = "/datasets/classname"

DetectorFactory.seed = 0

with open(class_path) as f:
	classes = f.readlines()

try:
	client = ClipClient(
		url="https://knn.laion.ai/knn-service",
		indice_name="laion5B-L-14",
		aesthetic_score=9,
		aesthetic_weight=0.5,
		modality=Modality.IMAGE,
		num_images=100,
	)
except:
	client = ClipClient(
		url="https://knn.laion.ai/knn-service",
		indice_name="laion5B-L-14",
		aesthetic_score=9,
		aesthetic_weight=0.5,
		modality=Modality.IMAGE,
		num_images=80,
	)

for i in range(0, len(classes)):
	
	class_name = classes[i].rstrip()
	caption_file = class_name + '.txt'
	caption = []
	print(class_name)
	with open(os.path.join(caption_path, caption_file), 'a+') as f:
		for name in sorted(os.listdir(os.path.join(img_folder_path, class_name))):
			caption = []

			try:
				results = client.query(image = os.path.join(img_folder_path, class_name, name))
			except json.decoder.JSONDecodeError:
				print("error!!!")
				print(name)
			for i, line in enumerate(results):
				caption.append(line['caption'])
	
			for c1 in caption:
				try:
					if detect(c1) == 'en':
						f.write(c1 + '\n')
				except LangDetectException:
					pass
	print('finish!\n')