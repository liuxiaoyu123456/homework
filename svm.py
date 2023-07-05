import cv2
import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# 颜色特征提取
def color_feature(image):
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算颜色直方图
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # 归一化直方图
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 形状特征提取
def shape_feature(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    # 计算轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 计算轮廓特征
    perimeter = cv2.arcLength(contours[0], True)
    area = cv2.contourArea(contours[0])
    rect = cv2.minAreaRect(contours[0])
    width, height = rect[1]
    aspect_ratio = float(width) / height if height != 0 else 1
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 1
    # 返回轮廓特征向量
    return np.array([perimeter, area, aspect_ratio, circularity])

# 纹理特征提取
def texture_feature(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度共生矩阵
    glcm = greycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    # 计算灰度共生矩阵特征
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    # 返回纹理特征向量
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

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
        images.append(image)
# 提取图像特征向量
X_color = [color_feature(image) for image in images]
print(X_color)
X_shape = [shape_feature(image) for image in images]
print(X_shape)
X_texture = [texture_feature(image) for image in images]
print(X_texture)
X = np.concatenate([X_color, X_shape, X_texture], axis=1)
y = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练SVM分类器
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# 预测测试集并计算准确率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")