# Machine Learning

## 1) MachineLearning.py

- 선형대수학 파이썬

- Bayesian Concept Learning :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ae.gif)   ,    ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09b0.gif)  

  ​                                                                        posterior Likelihood prior



## 2) LinearRegression.py

- Regression(회귀) : 데이터의 경향성으로 연속적인 수치를 예측 vs Classification(분류) : 데이터를 정해진 범주에 따라 분류

   → Regression은 반응 변수가 연속적이라는 점을 제외하면 Classification과 같습니다.

  Linear Regression Model :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09b4.gif)  

  RSS (Residual Sum of Squares) :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09b6.gif)   → loss function, cost function으로 사용

  MSE (Mean Square) :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09b8.gif)  

  Finding the MLE :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ba.gif)   → Normal Equation :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09bc.gif)   , Ordinary Least Square :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09be.gif)  

  Gradient Descent Method :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09c0.gif)  Learning Rate)

  Convexity :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09c2.gif)  

  Stochastic Gradient Descent : Only using randomly picked one xi per step

  Linear Regression with Gaussian Distribution Likelihood :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09c4.gif)  

  Log-Likelihood :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09c6.gif)  

  Laplace Distribution :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09c8.gif)     (  ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ca.gif)  ) → Outlier가 가우시안 분포보다 더 잘 나옴

  Regularization : Overfitting 문제를 해결하기 위해 매개변수 앞에 가우시안을 추가하여 크기를 작게 하는 것

   → ridge regression (penalized least squares) :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09cc.gif)  



## 3) LogisticRegression.py

- NLL (Negative Log-Likelihood) :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09d0.gif)  

  Sigmoid Function :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09d2.gif)  

  **Q. Logistic Regression****에서** **Sigmoid** **함수를 사용하는 이유****?** 

  Bernoulli Distribution :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09d4.gif)                   → p(y|**x**, **w**) = Ber(y|sigm(**w**T**x**))

    ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09d6.gif)  

    ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09d8.gif)  

  Convexity :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09da.gif)   → Hessian Matrix   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09dc.gif)   → 모든 요소가 0보다 크면 Convex하다



## 4) Classification.py

## 5) Classification2.py

## 6) kNNclassification.py

- k-NN : k 개의 가장 가까운 이웃이 어떻게 구성되어 있는지 비교하여 분류를 수행한다.

  step 1. 입력에서 훈련 세트 데이터까지의 거리 계산

  ​     \2. 가장 가까운 이웃 거리의 "k"를 검사

  ​     \3. class의 대다수에 따라 class를 결정한다.

- Supervised Learning : 입력과 출력 데이터에 의해 구동된다 “training 세트“

  Unsupervised Learning : 입력 데이터에 의해서만 구동된다.

  Clustering : 대표적인 Unsupervised Learning 알고리즘

  k-Means Clustering Algorithm

   Initialize Z={z1, z2, ..., zk}

   while (true)

  ​    for (i=1 to N)                                        // M step (Maximization)

  ​       Map xi into the nearest zj

  ​    if (No change of mapping from the previous loop)

  ​       break

  ​    for (j=1 to K)                                        // E step (Expectation)

  ​       replace zj with the mean of the xi mapped to zj

   for (j=1 to K)

  ​    allocate the samples mapped to zj to cj

  Categorial Distribution : 여러 개의 값을 가질 수 있는 독립 확률 변수들에 대한 확률분포

  Mixture Model :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09e4.gif)   , prior :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09e6.gif)   , likelihood :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09e8.gif)  

  Gaussian Mixture Model :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ea.gif)    → 가우시안 분포가 여러 개 혼합된 Clustering 모델

  Soft Clustering : 각   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ec.gif)  에 대해 확률을 다 내는 것 vs Hard Clustering : 최대 확률을 가지는   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09ee.gif)  에 대해서만 내는 것

  GMM Clustering E step :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09f0.gif)   , M step :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09f2.gif)  

- Transforming Data : Increasing Dimension

  Linearizing Decision Boundary : D-차원 벡터로의 적절한 변환으로, 데이터는 더 높은 Q-차원에서 선형으로 분리 가능

  Kernel Function :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09f6.gif)   고차원에서의 벡터 계산을 저차원에서 계산으로 바꿔주는 역할

  RBF (Radial Basis Function) kernel :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09f8.gif)   , Polynomial kernel :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09fa.gif)  

  Kernel Trick for Ridge Regression :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c09fc.gif)  

- Lagrange Multiplier : Minimize f(x,y)=ax+by subject to g(x,y)=x2+y2-r →   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a00.gif)   ⇒   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a02.gif)       KKT↓

  Handling Inequality Constraints : Minimize f(x) subject to g(x)  ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a04.gif)  0 →   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a06.gif)   ⇒   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a08.gif)   ,   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a0a.gif)  



## 7) SupportVector.py

## 8) 머신러닝_HW_2015706035_\_홍세준.py

- Support Vector : 경계를 결정하는 샘플들

  Support Vector Machine : 마진을 최대화하여 일반적으로 분류 모델의 정확성을 향상시키는 것

  Boundary hyperplane : h(**x**)=**w**T**x**+b=0 , h(**x**)>0 → yi=+1 / h(**x**)<0 → yi=-1

  Distance Point to Hyperplane : (a1, a2, a3) ~ (b1, b2, b3)T  ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a0e.gif)  (x1, x2, x3)=c → a1b1+a2b2+a3b3-c ⇒   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a10.gif)  

  SVM apply Lagrange : Minimize   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a12.gif)   subject to   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a14.gif)  

   ⇒   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a16.gif)   ,   KKT :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a18.gif)  

   ⇒ Lagrange Auxiliary Function   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a1a.gif)   ,     ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a1c.gif)   ,     ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a1e.gif)  

  Hard Margin   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a20.gif)   조건만 갖는 경우

  Soft Margin   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a22.gif)   최소화 &   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a24.gif)   최소화                 ⇒   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a26.gif)  

  Minimize   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a28.gif)   subject to   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a2a.gif)  

- Training for Perceptron : cost function   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a2e.gif)   →   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a30.gif)  Multi Layer Perceptron (MLP) :   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a32.gif)   cost function   ![img](file:///C:\Users\msi\AppData\Local\Temp\DRW00005c1c0a34.gif)  