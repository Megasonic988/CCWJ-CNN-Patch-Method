import pyrebase
import cv2
import os
import numpy as np
import re
import sys

'''
Download the image and place them in the same directory as this file.
Run the script with the filename of the image as a command line argument:
python3 process_image.py <filename>
This program will download the image annotations from Firebase, then 
divide the image into patches. The patches are saved as <filename>_labels.npy 
and the labels as <filename>_subimages.npy.
'''

if len(sys.argv) == 1:
    exit('Please enter the filename of the image:\n python3 process_image.py <filename>')
FILE_NAME = sys.argv[1]
IMAGE_NAME = re.sub('[^a-zA-Z0-9]', '', FILE_NAME)

config = {
    "apiKey": "AIzaSyCkfjzqiYMmKMYi2whd3ixnWzcPi-vJu-Q",
    "authDomain": "ccwj-image-annotation.firebaseapp.com",
    "databaseURL": "https://ccwj-image-annotation.firebaseio.com",
    "projectId": "ccwj-image-annotation",
    "storageBucket": "ccwj-image-annotation.appspot.com",
    "messagingSenderId": "445077046053"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("user@example.ca", "Password")
token = user['idToken']
db = firebase.database()

# Get the labels for an image
ref = db.child('annotations').child(IMAGE_NAME)
data = ref.get(token).val()
radius = int((data[1]['x'] - data[0]['x']) / 2)

# Process image
if not os.path.exists('images'):
    os.makedirs('images')
img = cv2.imread(FILE_NAME)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
subimages = []
labels = []
for d in data:
    label = d['label']
    if label == 'Undefined':
        continue
    labels.append(label)
    y_min = d['y'] - radius
    y_max = d['y'] + radius
    x_min = d['x'] - radius
    x_max = d['x'] + radius
    subimage = img[y_min:y_max, x_min:x_max]
    subimages.append(subimage)

subimages = np.array(subimages)
np.save('images/' + IMAGE_NAME + '_subimages', subimages)
np.save('images/' + IMAGE_NAME + '_labels', labels)