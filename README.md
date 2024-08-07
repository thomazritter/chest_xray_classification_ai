# Chest X-ray Classification
 Classificação de imagens de exames de Raio-x torácico utilizando CNN e Resnet-50

## Autores 
 Nomes: Thomaz Justo, Yuri Kiefer Alves, Lucas Schneider e Thomas Sponchiado

## Introdução

 Este relatório apresenta o desenvolvimento de um pipeline de aprendizado de máquina para a tarefa de classificação de imagens utilizando redes neurais convolucionais (CNNs) e Resnet-50. Foi escolhido um conjunto de dados onde temos a radiografia de pacientes com COVID-19.

 O objetivo deste trabalho é explorar as técnicas de aprendizado de máquina e compreender os desafios inerentes à classificação de imagens. 

 Durante o processo, foram realizadas etapas de modelagem dos problemas, implementação das CNN & Resnet-50 utilizando o Keras, treinamento das redes neurais com os respectivos conjuntos de dados, otimização de hiperparâmetros e análise dos resultados obtidos.

## Metodologia

 Para desenvolver o pipeline de aprendizado de máquina, algumas etapas foram seguidas:

 - escolha dos datasets e seus pré-processamentos
 - definição inicial da arquitetura da rede
 - definição das funções de ativação e de erro
 - escolha dos otimizadores.

## Dataset

 Para treinar as redes neurais foi utilizado o dataset "COVID-19 Radiography Database", encontrado no repositório do Kaggle (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), que possuí 21165 imagens com resolução de 299x299 pixels e possui como features as próprias imagens de raio-x e uma máscara de segmentação somente para os pulmões em cada imagem.

 Os rótulos utilizados pelo dataset para classificar as imagens são: COVID, Lung_Opacity, Viral Pneumonia e Normal. Esta separação está inserida no nome do arquivo de cada imagem e este será o objetivo de classificação da CNN e o modelo Resnet-50 utilizados.

## Setup

 - Para utilizar o algoritmo é necessário realizar o download do dataset, disponível em: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
 - Mova o arquivo .zip baixado para a pasta do arquivo .py, para ser acessado pelo algoritmo
 - Caso não instaladas, instale as seguintes dependências:
 ```sh
 keras
 numpy
 matplotlib
 sklearn
 os
 cv2
 seaborn
 ZipFile
 tensorflow
 ```

## Modelos utilizados

 ### Pré processamento de dados

  A entrada dos dados no modelo pode ser realizada com os pixels da imagem ou features retiradas das imagens.

  Nesse caso foram usados apenas os pixels da imagem para a entrada das redes neurais.

  O pré-processamento realizado deixo todas as imagens em escala de cinza, diminuindo a complexidade do processamento da rede de 3 canais para um canal e todas foram redimensionadas para 224x224 pra servir tanto na CNN como na Resnet-50 pré-treinada.

  As classes foram retiradas diretamente da imagem, por meio da pasta que estão presentes e ocorreu um pré-processamento dos labels, que foram substituídos por números, para que pudessem ser usados nas redes neurais.

 ### Coleta de imagens do dataset

  As imagens do dataset foram organizadas em pastas, cada uma contendo as fotos da respectiva patologia, portanto é necessário retirar estas imagens das pastas e inseri-las em uma lista, onde poderão ser reorganizadas e receberão a classe, para sua futura classificação.

 ### Coleta de classes por meio das imagens

  Após a organização do dataset, este pode ser separado em teste e treino para que possa ser geradas as métricas de forma correta posteriormente.

  As classes das imagens foram convertidas de strings para números de forma que a CNN possa identificar estes labels.

 ### Randomização dos indices

  Para melhor separar o dataset em treino e teste, os índices das imagens e classes foram reorganizados de forma aleatória e após isso foram separados os dados em treino e teste para serem testados na CNN e Resnet

## Arquitetura de rede CNN

### Entrada de dados

Redes neurais convolucionais, ou CNNs, são algoritmos de Deep Learning muito utilizados para classificação e reconhecimento de imagens com pouco ou em alguns casos nenhum pré-processamento.

Assim como MLPs, são formados por neurônios que captam e processam a informação da camada anterior e repassam para a próxima camada de neurônios, porém o modelo escaneia as imagens com um kernel e realiza a convolução dos valores.

No caso da rede utilizada, a entrada de dados serão os pixels de cada imagem em formato 224x224 em escala de cinza.

### Camadas de convolução

As camadas seguintes são camadas de convolução, ocultas, que irão extrair features e padrões da imagem de entrada a partir do uso dos kernels (filtros), que realizam o produto escalar dos valores da imagem para formar uma nova imagem.

As CNN possuem uma certa arbitrariedade no número de camadas de convolução utilizada, portanto este valor pode ser ajustado de forma a se adequar melhor ao problema proposto.

### Camadas de pooling

Em seguida existem camadas de pooling, que tem como objetivo diminuir a quantidade de informação da imagem para simplificar seu processamento.
Os métodos utilizados podem ser por Max pooling, quando é escolhido o maior ponto da vizinhança para ser representado na imagem reduzida, ou o Average pooling, quando é realizada a média dos valores da vizinhança e este valor será representado na imagem simplificada.

### Camada MLP de classificação

No final do modelo, camadas de neurônios totalmente conectados (MLPs) são utilizados para classificar a imagem e apresentar a classificação encontrada pelo modelo.

### Funções de ativação

Utilizado para algoritmos complexos por conta da sua não-linearidade, onde algoritmos lineares não seriam capazes de realizar a tarefa.

São funções que realizam operações matemáticas ativando ou não neurônios de uma camada com base nas informações de camadas anteriores, fazendo com que a saída seja ligeiramente diferente da época anterior mas sem perder os resultados bons gerados pela alteração dos pesos dos neurônios.

As funções mais utilizadas são: Linear, Sigmoid, Tanh,  ReLU e Softmax

### Funções de erro

São responsáveis por medir as diferenças entre os resultados previstos pelo mmodelo e os resultados reais adquiridos a partir do dataset.

Funções como categorical crossentropy ou sparse categorical crossentropy são utilizados para problemas multiclasse, enquanto binary crossentropy é utilizado para categorias binárias e o erro quadrático ou absoluto é usado para problemas de regressão.

Para funções multiclasse essa perda é calculada com base no número de referências no lote (batch) e a quantidade de respostas corretas que o modelo previu

### Otimizadores

São algoritmos que auxiliam o processo de aprendizado ajustando os pesos dos neurônios nas camadas do modelo e podem alterar o coeficiente de aprendizado, para que se torne mais lento conforme converge ao ponto ótimo, impedindo que este ultrapasse o ponto ideal e gere novos erros.

Alguns exemplos são: Adam, Adagrad e RMSprop que possuem mesma ideia de alterar a taxa de aprendizado conforme os gradientes passados porém com pequenas diferenças no modo colo lidam com taxas de aprendizado muito pequenas.

Exemplos como SDG e Momentum atualizam os pesos com base os gradientes porém de maneira também um pouco diferente, tornando um mais rápido que o outro, porém aumentando os ruídos.

## Treinamento do modelo CNN

No treinamento da CNN, o modelo recebe as imagens no formato pré-determinado, realiza as convoluções com cada imagem e para tomada de decisão possui um número de referências aos quais vai basear sua escolha.

O processo é repetido pelo número de épocas determinado no início do teste e a cada época o valor dos pesos dos neurônios são ajustados pelo otimizador, também determinado ao início do teste, para reduzir as perdas do modelo na predição de imagens.

Foram realizados testes para cada rede, onde são exploradas no mínimo três variações de: número de camadas, quantidade de neurônios por camada, tipos de funções de ativação, funções de erro e otimizadores. Dessa maneira, planeja-se executar no mínimo 30 experimentos distintos.

### Resultados

Obtivemos um bom resultado utilizando a rede CNN, sendo a acurácia de 93% e o loss de 0.33

## Arquitetura de rede Resnet-50

A partir do modelo de CNN pode ser inferido outros modelos, como a Resnet-50 (que vem do nome Residual Network) e que é bem semelhante a ultima apresentada, com exceção do número de camadas de convolução, que nesse caso são 48 camadas, que somadas a uma camada de Max pooling e uma camada de Average pooling formam 50 camadas, dando nome ao modelo.

Este modelo é o sucessor da rede Resnet-34 (já sucessora do modelo VGG 16 e 19), que como o nome diz, possuia 34 camadas com neurônios de pesos diferentes e possuia a capacidade de pular camadas de neurônios quando necessário e assim se tornando uma rede residual e não mais uma rede convolucional padrão.

A rede possui 2 regras de design. A primeira diz que o número de filtros em cada camada é o mesmo e a segunda diz que se o número do mapa de features cair pela metade, os filtros devem compensar dobrando de tamanho, para manter a complexidade no tempo de cada camada.

Para auxiliar o processo de aprendizado da rede, esta pode ser importada de forma pré-treinada por datasets padrões e apenas ser realizados ajustes nas camadas finais ou um fine tuning do modelo.

O modelo para classificação das imagens do dataset utilizado no trabalho foi pré-treinado com as imagens do dataset Imagenet e para que as imagens possam ser utilizadas em escala de cinza, como o modelo acima, a camada inicial do modelo foi substituida por uma que aceite imagens com apenas um canal de cor. Para classificação foi adicionado uma camada densa ao final do modelo, onde será realizado o fine tuning da Resnet.

## Treinamento Resnet-50

Assim como a CNN o treinamento ocorre com base no número de épocas, a quantidade de referências para tomada de decisão e os diferentes otimizadores, que foram testados também com mais e 30 diferentes configurações.

### Resultados

Buscamos um modelo com melhor acurácia utilizando um número baixo de épocas e encontramos um modelo com resultados acima de 90.9% de acurácia e 0.24 de loss.
Ao utilizar as 50 épocas, nao obtivemos um resultado melhor do que o original. Obtivemos aqui 90.6% de acurácia e 0.34 de loss.

## Conclusão & Referências & Vídeo

A utilização de CNNs para classificação de imagens do dataset apresentou eficácia na classificação tanto em função do tempo quanto acurácia, chegando a mais de 90% de acurácia em 10 épocas, com um loss abaixo de 0.4 no dataset de teste.

Em comparação com a Resnet-50, após ajustes a CNN apresentou resultado melhor em questão de precisão dos resultados e diminuição do loss, entretanto é necessário maior quantidade de treinamento, que em situações mais complexas ou com maior quantidade de imagens para treinamento pode se tornar um gargalo para o sistema e um atraso na entrega de projetos.

A Resnet cumpriu seu objetivo de forma eficiente por necessitar de pouco treinamento, inferindo que o pré-treinamento da rede mesmo que com imagens totalmente diferentes é benéfico para criar filtros genéricos e entender padrões de forma mais rápida, chegando próximo a 90% de acerto já em 2 épocas, porém sendo necessário uma quantidade muito grande de épocas para chegar a 93% de acurácia como o melhor modelo apresentou.

Um problema que pode ser notado é a dificuldade na classificação de algumas imatgens. A hipótese é que estes erros ocorreram devido a grande diferença no número de amostras do dataset, onde quase metade deste se apresenta em apenas uma classe. Entretanto esse problema não interferiu de maneira tão grave no modelo a ponto de não ser possível realizar a classificação

Vídeo de apresentação: https://youtu.be/jJI6fs9V7w4?si=zC1o56KrbsCVcJnv

## Referências:

CNN:

https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

https://keras.io/examples/vision/mnist_convnet/

https://keras.io/examples/vision/

Resnet-50:

https://datagen.tech/guides/computer-vision/resnet-50/

https://keras.io/api/applications/resnet/

## Meta

Thomas Sponchiado Pastore – [@Thomas_spastore](https://www.instagram.com/thomas_spastore?igsh=Z3RlYjRjaThpNmlu) – thomas.spastore@gmail.com

Thomaz Justo - [@thomaz_justo](https://www.instagram.com/thomaz_justo?igsh=MWE5aDN5enh2Y2E3aw==)

Yuri Kiefer Alves - [@poraiyuri](https://www.instagram.com/poraiyuri?igsh=MTQxdmN1aGNjczEwdQ==)

Lucas Schneider - schneider.lusca@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/thom01s](https://github.com/thom01s?tab=repositories/)

## Contribuição
1. Fork it (<https://github.com/thom01s/Chest-X-ray-Classification/fork>)
3. Create your feature branch
4. Commit your changes
5. Push to the branch
6. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
