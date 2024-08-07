"""
Nomes: Thomaz Justo, Yuri Kiefer Alves, Lucas Schneider e Thomas Sponchiado

# Introdução
Este relatório apresenta o desenvolvimento de um pipeline de aprendizado de máquina para a tarefa de classificação de imagens utilizando redes neurais convolucionais (CNNs) e Resnet-50. Foi escolhido um conjunto de dados onde temos a radiografia de pacientes com COVID-19.
O objetivo deste trabalho é explorar as técnicas de aprendizado de máquina e compreender os desafios inerentes à classificação de imagens. Durante o processo, foram realizadas etapas de modelagem dos problemas, implementação das CNN & Resnet-50 utilizando o Keras, treinamento das redes neurais com os respectivos conjuntos de dados, otimização de hiperparâmetros e análise dos resultados obtidos.

# Metodologia
Para desenvolver o pipeline de aprendizado de máquina, algumas etapas foram seguidas:
- escolha dos datasets e seus pré-processamentos
- definição inicial da arquitetura da rede
- definição das funções de ativação e de erro
- escolha dos otimizadores.

# Dataset
Para treinar as redes neurais foi utilizado o dataset "COVID-19 Radiography Database", encontrado no repositório do Kaggle (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), que possuí 21165 imagens com resolução de 299x299 pixels e possui como features as próprias imagens de raio-x e uma máscara de segmentação somente para os pulmões em cada imagem.
Os rótulos utilizados pelo dataset para classificar as imagens são: COVID, Lung_Opacity, Viral Pneumonia e Normal. Esta separação está inserida no nome do arquivo de cada imagem e este será o objetivo de classificação da CNN e o modelo Resnet-50 utilizados.

### COVID-19 Radiography Database
O dataset escolhido contém um banco de imagens de raios-X de tórax para casos positivos de COVID-19, casos normais, casos de Lung Opacity e por fim, casos de Pneumonia Viral. Conjunto de exemplo abaixo:

# Pré processamento de dados
A entrada dos dados no modelo pode ser realizada com os pixels da imagem ou features retiradas das imagens.
Nesse caso será usado apenas os pixels da imagem para a entrada das redes neurais.
O pré-processamento realizado deixa todas as imagens em escala de cinza, diminuindo a complexidade do processamento da rede de 3 canais para um canal e todas foram redimensionadas para 224x224 pra servir tanto na CNN como na Resnet-50.
As classes foram retiradas diiretamente da imagem, por meio da pasta que estão presentes e ocorreu um pré-processamento dos labels, que foram substituídos por números, para que pudessem ser usados nas redes neurais.

# Coleta de imagens do dataset
As imagens do dataset foram organizadas em pastas, cada uma contendo as fotos da respectiva patologia, portanto é necessário retirar estas imagens das pastas e inseri-las em uma lista, onde poderão ser reorganizadas e receberão a classe, para sua futura classificação.
As imagens são transformadas para ter cor em escala de cinza e redimensionadas para poderem ser usadas da mesma forma em ambos os modelos testados
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn import metrics
import seaborn as sns
import ZipFile
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, callbacks

# Descompacta o .zip baixado do dataset
radiographyZip = ZipFile('covid19-radiography-database.zip', 'r')
radiographyZip.extractall(path='')

def load_images(pastas):
  images = []
  filename = []
  for pasta in pastas:
      caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
      arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
      imgs = [arq for arq in arquivos if arq.lower().endswith(".png")]
      filename.append(imgs)

      for file_name in imgs:
          img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
          img = cv2.resize(img, (224, 224))
          images.append(img)
  return images, filename

X, y = load_images(['COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/COVID/images',
          'COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Lung_Opacity/images',
          'COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Normal/images',
          'COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Viral Pneumonia/images'])
y = [item for sublista in y for item in sublista]

"""
#Coleta de classes por meio das imagens

Após a organização do dataset, este poderá ser separado em teste e treino para que possa ser geradas as métricas de forma correta posteriormente.
As classes das imagens foram convertidas de strings para números de forma que a CNN possa identificar estes de forma fácil.
"""

for i in range(0, len(y)):
    if 'COVID/images' in y[i]:
        y[i] = 0
    elif 'Lung_Opacity/images' in y[i]:
        y[i] = 1
    elif 'Normal/images' in y[i]:
        y[i] = 2
    elif 'Viral Pneumonia/images' in y[i]:
        y[i] = 3
        
"""
# Randomização dos indices
Para melhor separar o dataset em treino e teste, os índices das imagens e classes foram reorganizados de forma aleatória e após isso foram separados os dados em treino e teste para serem testados na CNN e Resnet
"""

emparelhada = list(zip(X, y))

np.random.shuffle(emparelhada)
X, y = zip(*emparelhada)

X = list(X)
y = list(y)

X = np.array(X)
y = np.array(y)

#separar em treino e teste de forma aleatória
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

#numero de categorias a classificar
num_classes = len(np.unique(y))

"""
# Arquitetura de rede CNN

## Entrada de dados
Redes neurais convolucionais, ou CNNs, são algoritmos de Deep Learning muito utilizados para classificação e reconhecimento de imagens com pouco ou em alguns casos nenhum pré-processamento.
Assim como MLPs, são formados por neurônios que captam e processam a informação da camada anterior e repassam para a próxima camada de neurônios, porém o modelo escaneia as imagens com um kernel e realiza a convolução dos valores.
No caso da rede utilizada, a entrada de dados serão os pixels de cada imagem em formato 224x224 em escala de cinza

## Camadas de convolução
As camadas seguintes são camadas de convolução, ocultas, que irão extrair features e padrões da imagem de entrada a partir do uso dos kernels (filtros), que realizam o produto escalar dos valores da imagem para formar uma nova imagem.
As CNN possuem uma certa arbitrariedade no número de camadas de convolução utilizada, portanto este valor pode ser ajustado de forma a se adequar melhor ao problema proposto.

## Camadas de pooling
Em seguida existem camadas de pooling, que tem como objetivo diminuir a quantidade de informação da imagem para simplificar seu processamento. Os métodos utilizados podem ser por Max pooling, quando é escolhido o maior ponto da vizinhança para ser representado na imagem reduzida, ou o Average pooling, quando é realizada a média dos valores da vizinhança e este valor será representado na imagem simplificada.

## Camada MLP de classificação
No final do modelo, camadas de neurônios totalmente conectados (MLPs) são utilizados para classificar a imagem e apresentar a classificação encontrada pelo modelo.

## Funções de ativação
Utilizado para algoritmos complexos por conta da sua não-linearidade, onde algoritmos lineares não seriam capazes de realizar a tarefa.
São funções que realizam operações matemáticas ativando ou não neurônios de uma camada com base nas informações de camadas anteriores, fazendo com que a saída seja ligeiramente diferente da época anterior mas sem perder os resultados bons gerados pela alteração dos pesos dos neurônios.
As funções mais utilizadas são: Linear, Sigmoid, Tanh,  ReLU e Softmax

## Funções de erro
São responsáveis por medir as diferenças entre os resultados previstos pelo mmodelo e os resultados reais adquiridos a partir do dataset.
Funções como categorical crossentropy ou sparse categorical crossentropy são utilizados para problemas multiclasse, enquanto binary crossentropy é utilizado para categorias binárias e o erro quadrático ou absoluto é usado para problemas de regressão.
Para funções multiclasse essa perda é calculada com base no número de referências no lote (batch) e a quantidade de respostas corretas que o modelo previu

## Otimizadores
São algoritmos que auxiliam o processo de aprendizado ajustando os pesos dos neurônios nas camadas do modelo e podem alterar o coeficiente de aprendizado, para que se torne mais lento conforme converge ao ponto ótimo, impedindo que este ultrapasse o ponto ideal e gere novos erros.
Alguns exemplos são: Adam, Adagrad e RMSprop que possuem mesma ideia de alterar a taxa de aprendizado conforme os gradientes passados porém com pequenas diferenças no modo colo lidam com taxas de aprendizado muito pequenas.
Exemplos como SDG e Momentum atualizam os pesos com base os gradientes porém de maneira também um pouco diferente, tornando um mais rápido que o outro, porém aumentando os ruídos.
"""
#%%
## Modelo completo

y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

def make_model(input_shape):

    #Camada inicial com entrada de dados
    input_layer = keras.layers.Input(shape=input_shape)

    #primeira cadama de convolução
    conv01 = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same")(input_layer)
    conv01 = keras.layers.BatchNormalization()(conv01)
    conv01 = keras.layers.ReLU()(conv01)

    pool01 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv01)

    #segunda cadama de convolução
    conv0 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same")(pool01)
    conv0 = keras.layers.BatchNormalization()(conv0)
    conv0 = keras.layers.ReLU()(conv0)

    pool0 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    #terceira cadama de convolução
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(pool0)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #quarta camada de convolução
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #quinta camada de convolução
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    #sexta camada de convolução
    conv4 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.ReLU()(conv4)

    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.ReLU()(conv5)

    #camada de pooling
    gap = keras.layers.GlobalAveragePooling2D()(conv5)

    #camada de flattening
    flatten = keras.layers.Flatten()(gap)

    #modelo MLP para classificação
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(flatten)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

"""
#Treinamento do modelo CNN

No treinamento da CNN, o modelo recebe as imagens no formato pré-determinado, realiza as convoluções com cada imagem e para tomada de decisão possui um número de referências aos quais vai basear sua escolha.
O processo é repetido pelo número de épocas determinado no início do teste e a cada época o valor dos pesos dos neurônios são ajustados pelo otimizador, também determinado ao início do teste, para reduzir as perdas do modelo na predição de imagens.
Serão realizados testes para cada rede, onde serão exploradas três variações de: número de camadas, quantidade de neurônios por camada, tipos de funções de ativação, funções de erro e otimizadores. Dessa maneira, planeja-se executar no mínimo 30 experimentos distintos.
"""

model = make_model(input_shape=(224, 224, 1))
keras.utils.plot_model(model, show_shapes=True)

epochs = 10
batch_size = 32

callbacks = [
    callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
    callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),]

model.compile(
    #optimizer= Adam(learning_rate = 0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
history = model.fit(
    X_train,
    #y_train,
    y_train_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = keras.models.load_model("best_model.keras")

#test_loss, test_acc = model.evaluate(X_test, y_test)
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
##Plot dos Resultados

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, fmt="d", linewidths=0.5, square=True, cmap="Blues")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.show()

"""
### Resultados

Obtivemos um bom resultado com a configuração acima, sendo a acurácia de 93% e o loss de 0.33
"""
#%%
"""
# Arquitetura de rede Resnet-50

A partir do modelo de CNN pode ser inferido outros modelos, como a Resnet-50 (que vem do nome Residual Network) e que é bem semelhante a ultima apresentada, com exceção do número de camadas de convolução, que nesse caso são 48 camadas, que somadas a uma camada de Max pooling e uma camada de Average pooling formam 50 camadas, dando nome ao modelo.
Este modelo é o sucessor da rede Resnet-34 (já sucessora do modelo VGG 16 e 19), que como o nome diz, possuia 34 camadas com neurônios de pesos diferentes e possuia a capacidade de pular camadas de neurônios quando necessário e assim se tornando uma rede residual e não mais uma rede convolucional padrão.
A rede possui 2 regras de design. A primeira diz que o número de filtros em cada camada é o mesmo e a segunda diz que se o número do mapa de features cair pela metade, os filtros devem compensar dobrando de tamanho, para manter a complexidade no tempo de cada camada.
Para auxiliar o processo de aprendizado da rede, esta pode ser importada de forma pré-treinada por datasets padrões e apenas ser realizados ajustes nas camadas finais ou um fine tuning do modelo.
O modelo para classificação das imagens do dataset utilizado no trabalho foi pré-treinado com as imagens do dataset Imagenet e para que as imagens possam ser utilizadas em escala de cinza, como o modelo acima, a camada inicial do modelo foi substituida por uma que aceite imagens com apenas um canal de cor. Para classificação foi adicionado uma camada densa ao final do modelo, onde será realizado o fine tuning da Resnet
"""

## Modelo Resnet-50 completo

def preprocess_input(img):
    return np.repeat(img[..., np.newaxis], 3, axis=-1)

# Supondo que X_train e X_val são seus dados de entrada
X_train_preprocessed = preprocess_input(X_train)
X_test_preprocessed = preprocess_input(X_test)

def make_model():
    # Carregar o modelo ResNet-50 pré-treinado, sem incluir a primeira camada
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))

    # Congelar as camadas da base_model
    base_model.trainable = False

    # Adicionar novas camadas no topo do modelo base
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')])
    return model
    
"""
#Treinamento Resnet-50

Assim como a CNN o treinamento ocorre com base no número de épocas, a quantidade de referências para tomada de decisão e os diferentes otimizadores, que serão testados a seguir.
"""

# Compilar o modelo
model.compile(optimizer= keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir callbacks
model_callbacks = [
    callbacks.ModelCheckpoint("best_model_resnet.keras", save_best_only=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)
]

# Treinar o modelo
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=2,
                    callbacks=model_callbacks,
                    validation_split=0.2,
                    verbose=1)

# Carregar o melhor modelo salvo
model = models.load_model("best_model_resnet.keras")

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#%%
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#%%
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, fmt="d", linewidths=0.5, square=True, cmap="Blues")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.show()

"""
### Resultados

Buscamos um dos modelos com melhor acurácia e encontramos um modelo com resultados acima de 90.9% de acurácia e 0.24 de loss.
Ao utilizar as 50 épocas, nao obtivemos um resultado melhor do que o original. Obtivemos aqui 90.6% de acurácia e 0.34 de loss.

# Conclusão & Referências & Vídeo

A utilização de CNNs para classificação de imagens do dataset apresentou eficácia na classificação tanto em função do tempo quanto acurácia, chegando a mais de 90% de acurácia em 10 épocas, com um loss abaixo de 0.4 no dataset de teste.
Em comparação com a Resnet-50, após ajustes a CNN apresentou resultado melhor em questão de precisão dos resultados e diminuição do loss, entretanto é necessário maior quantidade de treinamento, que em situações mais complexas ou com maior quantidade de imagens para treinamento pode se tornar um gargalo para o sistema e um atraso na entrega de projetos.
A Resnet cumpriu seu objetivo de forma eficiente por necessitar de pouco treinamento, inferindo que o pré-treinamento da rede mesmo que com imagens totalmente diferentes é benéfico para criar filtros genéricos e entender padrões de forma mais rápida, chegando próximo a 90% de acerto já em 2 épocas, porém sendo necessário uma quantidade muito grande de épocas para chegar a 93% de acurácia como o melhor modelo apresentou.
Um problema que pode ser notado é a dificuldade na classificação de algumas imatgens. A hipótese é que estes erros ocorreram devido a grande diferença no número de amostras do dataset, onde quase metade deste se apresenta em apenas uma classe. Entretanto esse problema não interferiu de maneira tão grave no modelo a ponto de não ser possível realizar a classificação

Vídeo de apresentação: https://youtu.be/jJI6fs9V7w4?si=zC1o56KrbsCVcJnv

Referências:

CNN:

https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
https://keras.io/examples/vision/mnist_convnet/
https://keras.io/examples/vision/

Resnet-50:

https://datagen.tech/guides/computer-vision/resnet-50/
https://keras.io/api/applications/resnet/
"""