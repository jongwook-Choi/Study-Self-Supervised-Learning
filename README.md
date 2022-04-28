# Study-Self-Supervised-Learning
> Self Supervised Representation Learning을 정리한 [원문](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)을 번역하며 공부를 시작합니다.

# Self-supervised Representation Learning
딥러닝과 머신러닝에서 충분한 레이블 정보가 주어진 지도학습(supervied learning)은 이미 매우 좋은 성능을 내고 있습니다. 좋은 성능은 일반적으로 많은 양의 레이블 정보가 필요하지만, 레이블링 작업은 비용 문제로 인해 규모를 키우기가 어렵습니다. 라벨링 되지 않은 데이터가 사람에 의해 라벨링 된 데이터에 비해 상당히 많은 것을 고려하면, 해당 데이터를 사용하지 않는 것은 다소 비효율적이라 생각됩니다. 하지만 레이블링이 안 된 데이터를 사용하는 비지도학습(unsupervised learning)은 지도학습에 비해 쉽지 않으며 훨씬 덜 효율적으로 작동합니다.

만약 레이블링 되지 않은 데이터에 대한 레이블을 무료로 얻을 수 있고 얻은 레이블을 바탕으로 지도학습 관점으로 학습이 가능하다면 어떨까요? 이를 위해 레이블링이 되어있지 않은 데이터에 대해 특별한 형태의 예측을 하도록 하는 지도학습을 정의해 학습하고, 학습한 모델의 정보를 일부분만 사용합니다. 우리는 이것을 **_self supervised leraning_** 이라고 합니다.  

이 아이디어는 자연어처리에서 이미 많이 사용되고 있었습니다. 자연어처리 모델의 기본 태스크는 과거의 시퀀스를 바탕으로 다음 단어를 예측하는 것 입니다. [BERT](https://arxiv.org/abs/1810.04805)는 스스로 생성한 레이블을 통해 정의된 두개의 유사 태스크를 추가했습니다. 


<p align='center'>
  <img src="./images/1.self-supervised-Yann-LeCun.png" alt/>
</p>
<p align='center'>
  <em>Fig. 1 self-supervised learning 구축 과정을 설명, [LeCun's Talk](https://www.youtube.com/watch?v=7I0Qt7GALVk)  </em>
</p>

다음 [저장고](https://github.com/jason718/awesome-self-supervised-learning)를 확인하면 잘 엄선된 self-supervised learning 연구 리스트를 확인할 수 있습니다. 깊이 있는 학습에 관심이 있다면 반드시 논문을 읽어보시기를 바랍니다.  

해당 포스트는 NLP와 Generative model에 대해 자세히 설명하지 않습니다.

# Why Self-Supervied Learning?
Self-supervised learning 데이터로부터 얻는 다양한 레이블 정보를 자유롭게 활용 가능하다는 장점이 있습니다. 해당 연구가 진행된 동기는 꽤 직관적입니다. 완벽히 레이블링 된 데이터셋을 만드는 것은 매우 큰 비용이 소모되지만, 레이블이 되어있지 않은 데이터는 매 순간 생성되고 있습니다. 이 더 많은 레이블링 되지 않은 데이터셋을 사용하기 위해, 데이터셋 자체의 객관적인 특성을 배우도록 지도학습을 진행하는 방법이 있습니다.  

*self-supervised learning*은 *pretext task*라고도 부르며 지도학습의 손실함수를 사용할 수 있게 합니다. 하지만, 여기서 새롭게 발명해낸 태스크에 마지막 성능은 보통 중요하지 않습니다. 대신 해당 태스크를 통해 학습된 중간 단계의 representation이 의미론적 또는 구조론적 의미를 지니거나, 실질적인 *downstream task*에 도움이 되기를 바랍니다.  

예를 들어 이미지를 무작위로 회전시킨 후 모델이 얼마만큼 이미지가 회전하였는지를 학습할 수 있습니다. 회전 예측 태스크가 종료되면 마치 보조 태스크를 대할 때처럼, 실질적인 정확도는 중요하지 않습니다. 하지만 모델이 실제 환경 태스크에 도움이 되는 높은 품질의 잠재 변수를 학습하기를 기대합니다. 실제 환경 태스크의 예시로, 매우 적은 레이블을 가지고 모델을 학습해야 하는 경우가 있습니다.  

포괄적으로 말하자면, 모든 Generative model은 self-supervised로 고려될 수 있습니다만, 목적이 서로 다릅니다. Generative model은 다양하고 실제와 유사한 이미지를 생성하는 것에 집중하지만, self-supervisd representation learning은 다양한 환경에서 도움이 될 만한 범용성 있는 좋은 특징을 추출하는 것을 목표로 합니다.  

# Image-Based










