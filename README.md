# Study-Self-Supervised-Learning
> Self Supervised Representation Learning을 정리한 [원문](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)을 번역하며 공부를 시작합니다.

# Self-supervised Representation Learning
딥러닝과 머신러닝에서 충분한 레이블 정보가 주어진 지도학습(supervied learning)은 이미 매우 좋은 성능을 내고 있습니다. 좋은 성능은 일반적으로 많은 양의 레이블 정보가 필요하지만, 레이블링 작업은 비용 문제로 인해 규모를 키우기가 어렵습니다. 라벨링 되지 않은 데이터가 사람에 의해 라벨링 된 데이터에 비해 상당히 많은 것을 고려하면, 해당 데이터를 사용하지 않는 것은 다소 비효율적이라 생각됩니다. 하지만 레이블링이 안 된 데이터를 사용하는 비지도학습(unsupervised learning)은 지도학습에 비해 쉽지 않으며 훨씬 덜 효율적으로 작동합니다.

만약 레이블링 되지 않은 데이터에 대한 레이블을 무료로 얻을 수 있고 얻은 레이블을 바탕으로 지도학습 관점으로 학습이 가능하다면 어떨까요? 이를 위해 레이블링이 되어있지 않은 데이터에 대해 특별한 형태의 예측을 하도록 하는 지도학습을 정의해 학습하고, 학습한 모델의 정보를 일부분만 사용합니다. 우리는 이것을 **_self supervised leraning_** 이라고 합니다.  

이 아이디어는 자연어처리에서 이미 많이 사용되고 있었습니다. 자연어처리 모델의 기본 태스크는 과거의 시퀀스를 바탕으로 다음 단어를 예측하는 것 입니다. [BERT](https://arxiv.org/abs/1810.04805)는 스스로 생성한 레이블을 통해 정의된 두개의 유사 태스크를 추가했습니다. 

![img1](./images/1.self-supervised-Yann-LeCun.png)
