import sys, os
sys.path.append(os.pardir)
from PIL import Image
import glob
import time

def load_images():
	data_dir = "../data/public/train"
	img_list = []

	for (root, dirs, files) in os.walk(data_dir):
		if len(files) > 0:
			for file in files:
				img = Image.open(root + '/' + file)
				img = img.resize((256,256))
				# img_list.append(img)
			tmp_files = glob.glob(root + '/*.jpg')
			for file in tmp_files:
				break			

	return img_list

if __name__ == "__main__":
	tmp_list = load_images()

	print(len(tmp_list))