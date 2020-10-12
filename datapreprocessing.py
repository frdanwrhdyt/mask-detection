import cv2
import os
import numpy as np
from keras.utils import np_utils


class DataPreprocessing:
    def __init__(self):
        self.data_path = 'dataset'
        self.categories = os.listdir(self.data_path)
        self.labels = [i for i in range (len(self.categories))]

        self.label_dict=dict(zip(self.categories, self.labels))
    
    def ShowDict(self):
        print(self.label_dict)
        print(self.categories)
        print(self.labels)


    def Train(self):
        data = []
        target = []
        img_size = 50
        for category in self.categories:
            folder_path = os.path.join(self.data_path, category)
            img_names = os.listdir(folder_path)

            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)

                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resize = cv2.resize(gray, (img_size, img_size))

                    data.append(resize)
                    target.append(self.label_dict[category])
                
                except Exception as e:
                    print('Exception : ', e)
        data = np.array(data)/255.0
        data = np.reshape(data,(data.shape[0],img_size,img_size,1))

        target = np.array(target)

        new_target = np_utils.to_categorical(target)
        np.save('data', data)
        np.save('target', new_target)
        print('Learning Complete')
        self.ShowDict()

if __name__ == '__main__':
    DataPreprocessing().Train()
