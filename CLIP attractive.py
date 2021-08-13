import torch
from PIL import Image
import requests
import numpy
from transformers import CLIPProcessor, CLIPModel
import glob
import csv
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


images=glob.glob("Images/*.jpg")


labels0 = ['white person', 'person of color']
labels1 = ['attractive', 'unattractive']
labels2 = ['rich', 'poor']
labels3 = ['friendly', 'unfriendly']
labels4 = ['intelligent', 'unintelligent']
labels5 = ['male', 'female']
for image in images:
 image=Image.open(image).resize((200, 200), Image.NEAREST)
 for i in range(6):
  globals()['inputs%s' % i] = processor(text=globals()['labels%s' % i], images=image, return_tensors="pt", padding=True)
  globals()['outputs%s' % i] = model(**globals()['inputs%s' % i])
  globals()['logits_per_image%s' % i] = globals()['outputs%s' % i].logits_per_image # image-text similarity score
  globals()['probs%s' % i]= globals()['logits_per_image%s' % i].softmax(dim=1) 
  globals()['data%s' % i]= globals()['probs%s' % i].detach().numpy()[0]
  with open("thisperson"+str(i)+".csv", "a") as fp:
   wr = csv.writer(fp, dialect='excel')
   wr.writerow(globals()['data%s' % i])
