import pickle
from flask import Flask, render_template, request
import os
from random import random
import cv2
import  sys
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
def TrichXuatFeatures(file):
  from tensorflow.keras.applications.vgg16 import VGG16
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.applications.vgg16 import preprocess_input
  model_VGG16 = VGG16(weights='imagenet', include_top=False)
  from keras.preprocessing.image import img_to_array, load_img  
  img = image.load_img(file, target_size=(224, 224)) # chuyển ảnh về size (224,224)
  x = image.img_to_array(img)        # chuyển ảnh về thành 1 array
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  x = preprocess_input(x)
  features = model_VGG16.predict(x)
  features = np.array(features).reshape(-1,1)
  return features
def DuDoan1AnhVGG16(model,file): # dự đoán 1 ảnh dựa theo cách trích xuất VGG16
  X = []
  features = TrichXuatFeatures(file)
  X.append(features)
  X = np.array(X)
  dimX1_, dimX2_, dimX3_ =X.shape
  X = np.reshape(np.array(X), (dimX1_, dimX2_*dimX3_))
  y_pred = model.predict(X)
  return y_pred
# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

filename = 'model.sav'
model_ = pickle.load(open(filename, 'rb'))



# Hàm xử lý request
@app.route("/", methods=['GET','POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         
            # Lấy file gửi lên
            image_ = request.files['file']
            
                # Lưu file
            dirs_image = os.listdir(app.config['UPLOAD_FOLDER'])
            print(image_.filename)
            print(app.config['UPLOAD_FOLDER'])
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image_.filename)
            for file_ in dirs_image:
              if file_ == image_.filename:
                res = DuDoan1AnhVGG16(model_,path_to_save)
                print(image_.filename)
                # return render_template("index.html", user_image = image.filename , result = res)
                return render_template("index.html",  msg = str(res[0]),user_image = image_.filename)
        
            
            print("Save = ", path_to_save)
            image_.save(path_to_save)                                        
            res = DuDoan1AnhVGG16(model_,path_to_save)
            return render_template("index.html",  msg = str(res[0]),user_image = image_.filename)


    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
    app.run()
      # app.run(host="0.0.0.0")
