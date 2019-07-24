from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

#加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape',train_images.shape)
print('len(train_labels)',len(train_labels))

print('test_images.shape',test_images.shape)
print('len(test_labels)',len(test_labels))


#网络架构

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#编译步骤
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#准备图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#准备标签


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#拟合模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#测试性能
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

'''
1）学习率

学习率（learning rate或作lr）是指在优化算法中更新网络权重的幅度大小。学习率可以是恒定的、逐渐降低的，基于动量的或者是自适应的。
不同的优化算法决定不同的学习率。当学习率过大则可能导致模型不收敛，损失loss不断上下震荡；学习率过小则导致模型收敛速度偏慢，需要更长的时间训练。通常lr取值为[0.01,0.001,0.0001]

（2）批次大小batch_size

批次大小是每一次训练神经网络送入模型的样本数，在卷积神经网络中，大批次通常可使网络更快收敛，
但由于内存资源的限制，批次过大可能会导致内存不够用或程序内核崩溃。bath_size通常取值为[16,32,64,128]

（3）优化器optimizer

目前Adam是快速收敛且常被使用的优化器。随机梯度下降(SGD)虽然收敛偏慢，但是加入动量Momentum可加快收敛，同时带动量的随机梯度下降算法有更好的最优解，即模型收敛后会有更高的准确性。通常若追求速度则用Adam更多。

（4）迭代次数

迭代次数是指整个训练集输入到神经网络进行训练的次数，当测试错误率和训练错误率相差较小时，可认为当前迭代次数合适；当测试错误率先变小后变大时则说明迭代次数过大了，需要减小迭代次数，否则容易出现过拟合。

（5）激活函数

在神经网络中，激活函数不是真的去激活什么，而是用激活函数给神经网络加入一些非线性因素，使得网络可以更好地解决较为复杂的问题。比如有些问题是线性可分的，而现实场景中更多问题不是线性可分的，若不使用激活函数则难以拟合非线性问题，测试时会有低准确率。所以激活函数主要是非线性的，如sigmoid、tanh、relu。sigmoid函数通常用于二分类，但要防止梯度消失，故适合浅层神经网络且需要配备较小的初始化权重，tanh函数具有中心对称性，适合于有对称性的二分类。在深度学习中，relu是使用最多的激活函数，简单又避免了梯度消失。
'''