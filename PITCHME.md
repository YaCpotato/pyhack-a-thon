---?color=#040000

## 強化学習を完全に理解したい

---?color=#040000
### 強化学習とは
@box[bg-gold](試行錯誤を通じて「価値を最大化するような行動」を学習する手法。正解の出力をそのまま学習すれば良いわけではなく、もっと広い意味での「価値」を最大化する行動を学習する)
---?color=#040000
@css[text-white fragment](とにかく動かしてみよう)
---?color=#040000
### CartPole
OpenAI Gymに用意されている倒立振子問題(バランスゲーム)
@img[fragment](./carpolesc.png)
---?color=#040000
OpenAI Gymより、dqn_cartpole.pyで学んでみた  
#### DQN(Deep Q Network)とは
@box[bg-gold](最適行動価値関数をニューラルネットを使った近似関数で求める手法)  
Qiita参照：https://qiita.com/ishizakiiii/items/5eff79b59bce74fdca0d#q-learning

---?color=#040000
コードをみていきましょう
---?color=#040000
各ライブラリのインポート  

```python
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
```

pip installするもの numpy,gym,keras,tensorflow,keras-rl(githubより)

---?color=#040000
環境の定義(学習対象の行動に対して状態の遷移先を決定し、報酬をあたえるもの)

```python
ENV_NAME = 'CartPole-v0'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
```

---?color=#040000
モデルの定義  

```python
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
```

---?color=#040000

```python
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
```

学習過程描画のためのメモリ確保？
ボルツマン選択によって確率から行動を推論していく
---?color=#040000
### ボルツマン選択とは
@box[bg-gold](ボルツマン分布を用い、評価の高い行動には最も高い選択確率が与えられ、他の行動には相応の重みをかけて確率的に行動する手法。  最初はランダム法で行動を選択していき、Tが少なくなるに連れてグリーディ法になっていく)
@img[fragment](./Boltsman.png)
---?color=#040000
### ランダム法とは
@box[bg-gold](ランダムに選ぶやつ)
---?color=#040000
### グリーディ法とは
@box[bg-gold](最適化問題を解くとき，計算の各段階で最も利益の大きい部分解を選んでいき，それらの部分解を組み合わせたものを最終的な解とする)
Tは時間と共に0に収束する関数とする
はじめ、Tが大きい時は、ランダム法で行動を選択し、小さくなるに連れてグリーディ法に近づいていく
---?color=#040000
<h2>うーん</h2>
<h2 class="fragment">なるほどわからん</h2>
---?color=#040000
ランダム法からグリーディ法に変わっていくボルツマン選択の勝手なイメージ
---?color=#040000
@img[](./first.png)
---?color=#040000
@img[](./second.png)
---?color=#040000
@img[](./third.png)
---?color=#040000
完全に理解した
---?color=#040000
---?color=#040000
---?color=#040000
