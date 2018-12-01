---
title: Model-based Novelty Detection
categories:
- General
excerpt: |
  Model-based Novelty Detection
feature_text: |
  ## Model-based Novelty Detection
  <br><br/>
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
use_math: true
---

Novelty detection이란, 데이터에서 outlier를 잡아내는 것을 말하며, 이 글에서는 모델을 이용해 outlier를 잡아내는 것에 대해 다루고자 합니다. Novelty detection에 사용될 수 있는 대표적인 모델로는 다음과 같이 3가지를 들 수 있습니다.<br/>

* Autoencoder
* One-Class SVM & SVDD
* Isolation Forest

<br/>
<h2> Auto-Encoder for Novelty Detection </h2>

Autoencoder는 neural network 모델의 한 종류이며, input data를 받았을 때 이와 똑같은 형태의 output data를 예측하는 것을 목적으로 합니다.  Autoencoder의 전체적인 모델 구조는 다음 사진과 같습니다.
<img src="/images/01_model_structure.png" width="1800" height="600" />
<br/>

Autoencoder는 크게 네 부분으로 나눌 수 있습니다. <br/>
* mapping layer (= encoder) 
* bottleneck layer
* demapping layer (=decoder)
* output layer

<br/>
  
encoder에서는 input data를 bottleneck layer로 보냄으로써 input 정보를 저차원으로 압축시킵니다.  <br/>
decoder에서는 압축된 형태의 input 정보를 원래의 input data로 복원하는 일을 수행합니다. <br/>

앞서 언급했듯이, 이 모델의 목표는 input data와 똑같은 형태의 data를 예측하는 것입니다. <br/>
따라서, output layer를 통해 나온 모델의 예측된 데이터와 실제 데이터의 차이를 loss function으로 정하고, 이 loss를 줄이는 방향으로 모델을 훈련시킵니다.
<img src="/images/02_loss_function.png" width="1800" height="600" />
<br/>

Autoencoder의 이러한 reconstruction error는 데이터의 outlier를 잡아낼 때 사용될 수 있습니다. <br/>
즉, autoencoder는 훈련 과정에서 훈련 데이터와 똑같은 형태의 데이터를 예측하기 위해 훈련 데이터의 일반적인 특징을 배웠을 것입니다. <br/>
그러므로, 일반적인 특징을 지닌 정상 데이터를 input으로 받았을 때는 이와 유사한 데이터를 쉽게 예측할 수 있으므로 reconstruction error가 낮은 반면, <br/>
다른 데이터와는 상이한 특징을 지닌 outlier를 input으로 받았을 때는 이와 유사한 데이터를 예측하기 어려워 reconstruction error가 높을 것입니다. <br/>
결국, reconstruction error가 특정 값보다 높은 경우에는 outlier, 그렇지 않은 경우에는 정상 데이터로 분류하는 방식으로 autoencoder를 통한 novelty detection을 구현할 수 있습니다.


<br/><br/>
<h4> 코드 </h4>
<br/>


```python
import os, sys
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
import tensorflow as tf



### Build autoencoder model
class Autoencoder():
    def __init__(self, MODEL_DIR, mini_batch_size, learning_rate, num_epoch, num_encoder_decoder_nodes, num_latent_nodes):
        self.MODEL_DIR = os.path.abspath(MODEL_DIR)
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.num_encoder_decoder_nodes = num_encoder_decoder_nodes
        self.num_latent_nodes = num_latent_nodes
        # Other params
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.MODEL_CKPT = os.path.join(self.MODEL_DIR, "model.ckpt")
        self.each_epoch_loss = 0.0
        return

    def __prepare_data(self):
        ### Import dataset
        # Take only 2 features from original 30-dimensional data
        self.X = datasets.load_breast_cancer().data[:,:2]
        # Change dtype into float32
        self.X = self.X.astype(np.float32)

        ### Shuffle data
        np.random.shuffle(self.X)

        ### Split data into train and test set
        num_X = self.X.shape[0]
        # Use 80% of data as train set
        self.num_train_X = np.int(num_X*0.8)
        self.train_X = self.X[:self.num_train_X, :]

        ### Mini batch index
        self.mini_batch_idx = 0
        return

    def __generate_mini_batch(self):
        # Generate data with mini batch size
        if self.mini_batch_idx >= self.num_train_X:
            self.mini_batch_idx = 0
        mini_batch_X = self.train_X[self.mini_batch_idx:self.mini_batch_idx+self.mini_batch_size, :]
        self.mini_batch_idx += self.mini_batch_size
        return mini_batch_X

    def __build_model(self):
        # Input (and target)
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        # Encoder layer
        # encoder_output shape: [None, self.num_encoder_decoder_nodes]
        encoder_output = tf.layers.dense(inputs=self.input,
                                         units=self.num_encoder_decoder_nodes,
                                         kernel_initializer=self.weight_initializer)
        # Latent layer
        # latent_output shape: [None, self.num_latent_nodes]
        latent_output = tf.layers.dense(inputs=encoder_output,
                                        units=self.num_latent_nodes,
                                        activation=tf.nn.sigmoid,
                                        kernel_initializer=self.weight_initializer)
        # Decoder layer
        # decoder_output shape: [None, self.num_encoder_decoder_nodes]
        decoder_output = tf.layers.dense(inputs=latent_output,
                                         units=self.num_encoder_decoder_nodes,
                                         activation=tf.nn.sigmoid,
                                         kernel_initializer=self.weight_initializer)
        # Output layer
        # logit_shape: [None, 2]
        self.logits = tf.layers.dense(inputs=decoder_output,
                                      units=2,
                                      kernel_initializer=self.weight_initializer)
        # Loss
        self.loss = tf.losses.mean_squared_error(labels=self.input, predictions=self.logits)
        # Global step
        self.global_step = tf.train.get_or_create_global_step()
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss, global_step=self.global_step)
        # Model saver
        self.saver = tf.train.Saver()
        return

    def train(self):
        # Create data
        self.__prepare_data()
        # Build model graph
        self.__build_model()
        # Run the model
        with tf.Session() as sess:
            # If the model exists, load it
            if tf.train.checkpoint_exists(checkpoint_prefix=self.MODEL_CKPT):
                LATEST_CKPT = tf.train.latest_checkpoint(checkpoint_dir=self.MODEL_DIR)
                self.saver.restore(sess, LATEST_CKPT)
                print('Latest checkpoint restored')
            # Create new model
            else:
                global_initializer = tf.global_variables_initializer()
                sess.run(global_initializer)
                print('New model created')
            # Train the model
            for each_epoch in range(self.num_epoch):
                for each_step in range(self.num_train_X):
                    # Get train data with mini batch size
                    mini_batch_X = self.__generate_mini_batch()
                    # Calculate loss and update the model
                    each_step_loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.input: mini_batch_X})
                    self.each_epoch_loss += each_step_loss
                print("Epoch: {}       Loss: {}".format(each_epoch+1, self.each_epoch_loss/self.num_train_X))
                self.each_epoch_loss = 0
                # Save the model every 10 epochs
                if (each_epoch+1) % 10 == 0:
                    self.saver.save(sess, self.MODEL_CKPT, global_step=self.global_step)
                    print('Model saved')
        return

    # Plot normal and novel data
    def __plot(self, predict_reconstruction_errors):
        predict_reconstruction_errors = np.array(predict_reconstruction_errors)
        # If reconstruction error is not over specific value, the data is normal
        normal_data_idx = np.where(predict_reconstruction_errors <= 45)
        # If reconstruction error is over specific value, the data is novel
        abnormal_data_idx = np.where(predict_reconstruction_errors > 45)
        normal_data = self.X[normal_data_idx]
        abnormal_data = self.X[abnormal_data_idx]
        plt.scatter(normal_data[:, 0], normal_data[:,1], c='b')
        plt.scatter(abnormal_data[:, 0], abnormal_data[:, 1], c='r')
        plt.show()
        return

    def predict(self):
        # Create data
        self.__prepare_data()
        # Build model graph
        self.__build_model()
        # Run the model
        with tf.Session() as sess:
            # Dummy list
            predict_reconstruction_errors = []
            # Load model
            LATEST_CKPT = tf.train.latest_checkpoint(checkpoint_dir=self.MODEL_DIR)
            self.saver.restore(sess, LATEST_CKPT)
            print('Latest checkpoint restored')
            # Test model
            for each_step in range(self.X.shape[0]):
                each_step_loss = sess.run(self.loss, feed_dict={self.input: self.X[each_step, :].reshape(-1, 2)})
                predict_reconstruction_errors.append(each_step_loss)
        # Plot the result
        self.__plot(predict_reconstruction_errors)
        return

# Load trained model and detect novel data
autoencoder = Autoencoder('./model', 32, 0.01, 50, 500, 100)
autoencoder.predict()
```
<br/>

autoencoder로 novelty detection을 한 결과는 다음과 같습니다.
<img src="/images/03_detection.png" width="1800" height="600" />


<br/><br/><br/>
<h2> One-class SupportVector Machine </h2>

One-class SVM은 다음과 같은 구도에서
<img src="/images/11_one_svm.png" width="1800" height="600" />
SVM 답게 margin을 최대화 하는 기본적인 골조를 갖습니다.
<img src="/images/12_one_svm.png" width="1800" height="600" />
그러나 막연하게 margin만 커지라고 지시할 경우, decision boundary가 무한히 음/양의 방향으로 발산할 것입니다. <br/>
이를 제어하기 위하여, decision boundary가 원점으로부터 양의 방향으로 최대한 멀어지라는 제약을 추가합니다. <br/>
이렇게 하면 음수쪽 발산 문제는 해결할 수 있지만 여전히 decision boundary가 무한히 양의 방향으로 발산할 가능성이 존재합니다.
<img src="/images/13_one_svm.png" width="1800" height="600" />
이를 해결하기 위하여, decision boundary의 밖으로 나가 클래스로 인정되지 않는 샘플들을 패널티로 정의합니다. <br/>
그리고 이 패널티가 최소화 되도록 즉, 가능한한 최소한의 샘플만이 decision boundary 밖으로 classified 되게끔 위치 제약을 추가합니다.
<img src="/images/14_one_svm.png" width="1800" height="600" />
끝으로 라그랑제 제약조건을 걸고, KKT 조건을 풀면
<img src="/images/15_one_svm.png" width="1800" height="600" />
다음과 같은 최적화 문제로 수렴됩니다.
<img src="/images/16_one_svm.png" width="1800" height="600" />
여기서 내적의 특성을 이용하여 kernel trick을 사용할 수 있게 되는데, 이러한 다차원 공간 매핑을 가능하게 해주는 커널에는
* Polynomial kernel
* MLP kernel
* RBF (gaussian) kernel
등이 있습니다.



<h4> RBF를 이용한 코드예시 </h4>

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as np
## 1 ############################################################################################
def closest_centroid_novelty_score(from_point, to_centroids):
    all_dists = [0]*len(to_centroids)
    for each_centroid_idx, each_centroid in enumerate(to_centroids):
        each_dist = np.sqrt(np.sum((from_point - each_centroid)**2))
        all_dists[each_centroid_idx] = each_dist
    novelty_score = np.min(all_dists)
    return novelty_score
## 2 ############################################################################################
input_digits, target_labels = load_digits(n_class=10, return_X_y=True)
num_data = 1000
input_digits = input_digits[:num_data]
target_labels = target_labels[:num_data]
k_means_output_obj = KMeans(n_clusters=10, n_jobs=4).fit(input_digits)
centroids = k_means_output_obj.cluster_centers_
novelty_scores = [0]*num_data
for outer_idx in np.arange(num_data):
    novelty_scores[outer_idx] = closest_centroid_novelty_score(input_digits[outer_idx], centroids)
## 3 ############################################################################################
print(novelty_scores)
# Read data and set some threshold
threshold = 35
outliers = [idx for idx, element in enumerate(novelty_scores) if element > threshold]
print(outliers)
```

<br/><br/><br/>
<h2> SupportVector Data Description </h2>

SVDD는 다음과 같은 구도에서
<img src="/images/21_svdd.png" width="1800" height="600" />
원의 크기를 최소화 하라는 기본적인 골조를 갖습니다.
<img src="/images/22_svdd.png" width="1800" height="600" />
그러나 막연하게 원의 크기를 최소화 하라고 지시할 경우, decision boundary가 무한히 작은 한 점으로 수렴해 버리고 말 것입니다. <br/>
이를 해결하기 위하여, decision boundary의 밖으로 나가 클래스로 인정되지 않는 샘플들을 패널티로 정의합니다. <br/>
그리고 이 패널티가 최소화 되도록 즉, 가능한한 최소한의 샘플만이 decision boundary 밖으로 classified 되게끔 위치 제약을 추가합니다.
<img src="/images/23_svdd.png" width="1800" height="600" />
끝으로 라그랑제 제약조건을 걸고, KKT 조건을 풀면
<img src="/images/24_svdd.png" width="1800" height="600" />
다음과 같은 최적화 문제로 수렴됩니다.
<img src="/images/25_svdd.png" width="1800" height="600" />
여기서 내적의 특성을 이용하여 kernel trick을 사용할 수 있게 되는데, 이러한 다차원 공간 매핑을 가능하게 해주는 커널에는
* Polynomial kernel
* MLP kernel
* RBF (gaussian) kernel
등이 있습니다.




<h4> RBF를 이용한 코드예시 </h4>

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
## 1 ############################################################################################
def reconstruction_loss_novelty_score(from_points, to_reconstructed_points):
    novelty_scores = np.sqrt(np.sum((from_points - to_reconstructed_points)**2, axis=1))
    return novelty_scores
## 2 ############################################################################################
input_digits, target_labels = load_digits(n_class=10, return_X_y=True)
num_data = 100
input_digits = input_digits[:num_data]
target_labels = target_labels[:num_data]
pca_obj = PCA()
projected_input_digits = pca_obj.fit_transform(input_digits)
reconstructed_input_digits = pca_obj.inverse_transform(projected_input_digits)
novelty_scores = reconstruction_loss_novelty_score(input_digits, reconstructed_input_digits)
## 3 ############################################################################################
print(novelty_scores)
# Read data and set some threshold
threshold = 9e-14
outliers = [idx for idx, element in enumerate(novelty_scores) if element > threshold]
print(outliers)
```
