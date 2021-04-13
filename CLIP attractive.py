import backprop
import glob
from PIL import Image
import csv
api_key = None
ic = backprop.ImageClassification(api_key=api_key)
images=glob.glob("thisperson/*.jpg")

labels1 = ['attractive', 'unattractive']

for image in images:
	Image.open(image).resize((200, 200), Image.NEAREST)
	results1 = ic(image, labels1)
	data1 = list(results1.items())
	with open("thisperson1.csv", "a") as fp:
		wr = csv.writer(fp, dialect='excel')
		wr.writerow(data1)



