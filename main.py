import streamlit as st
import cv2, tempfile, os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


if not os.path.exists("frames"):
    os.mkdir("frames")

if not os.path.exists("labelled_images"):
    os.mkdir("labelled_images")

from keras.applications.vgg16 import VGG16
model = VGG16()

st.title("ASSIGNMENT 2 OBJECT DETECTION USING VGG16")
st.write("Samantha Nyasha Mugari And Rumbidzai Dumbu")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
i=0

vid = st.file_uploader(("Upload Video"),type=['mp4','mkv', 'avi'])
if vid is not None:
	TempFile = tempfile.NamedTemporaryFile(delete=False)
	TempFile.write(vid.read())
	cap= cv2.VideoCapture(TempFile.name)

	stframe = st.empty()

	while 1:
		ret, frame = cap.read()

		if not ret:
			break
		image = cv2.imwrite('./frames/assign'+str(i)+'.jpg',frame)
		image = load_img('./frames/assign'+str(i)+'.jpg', target_size=(224,224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		pred = model.predict(image)
		label = decode_predictions(pred, top = 1)
		label = decode_predictions(pred)
		label = label[0][0]
		lbl = str('%s (%.2f%%)' % (label[1], label[2]*100))
		frame = cv2.putText(frame, lbl , (25, 60), font, fontScale=font_scale,color=(0,255,255), thickness=3)
		#frame = cv2.imwrite('./labelled_images/'+str(label[1])+str(i)+'.jpg',frame)

		stframe.image(frame)

location = []
object = []
for file in os.listdir('./frames/'):
  print(file)
  imagespath ='./frames/' + file
  image = load_img(imagespath, target_size=(224,224))
  image = img_to_array(image)
  image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  pred = model.predict(image)
  label = decode_predictions(pred, top = 1)
  object.append(label[0][0][1])
  location.append(imagespath)
  print(label)
  print()


cv2.destroyAllWindows()

srch = st.text_input("Search for an object")
def search(list, srch):
  for i in range(len(list)):
  	if list[i] == srch:
  		return location[i]
  	else:
  		print("NO OBJECT DETECTED")
a = search(list= object, srch='')
print(a)
