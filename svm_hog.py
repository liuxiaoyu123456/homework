import os

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

path = './images'
path_name = []
for root, dirs, files in os.walk(path):
    for dir in dirs:
        path_name.append(dir)
# 加载图像数据集
images = []
labels = []
for i in range(0, 21):
    for j in range(0, 100):
        if(j<10):
            shuzi = '0'+str(j)
        else:
            shuzi = str(j)
        filename = './images/'+path_name[i]+'/'+path_name[i]+shuzi+'.tif'
        labels.append(i + 1)
        image = cv2.imread(filename)
        image = cv2.resize(image, (64, 64))  # 将图像调整为相同的大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 将图像转换为灰度图像
        images.append(image)

images = np.array(images)
# 将类别标签保存为numpy数组文件
labels = np.array(labels)
np.save('labels.npy', labels)
np.save('data.npy', images)
# 读取训练数据和标签
data = np.load('data.npy')
labels = np.load('labels.npy')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 提取HOG特征
hog_features_train = []
hog_features_test = []
for img in X_train:
    hog_features_train.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'))
for img in X_test:
    hog_features_test.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'))

# 转换成numpy数组
hog_features_train = np.array(hog_features_train)
hog_features_test = np.array(hog_features_test)

# 训练SVM分类器
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(hog_features_train, y_train)

# 进行预测并计算准确率
y_pred = svm.predict(hog_features_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)