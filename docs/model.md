# ブラウン運動の数値シミュレーション

周期的に幅が変化するような2次元空間において、電気双極子の性質を持つ棒状粒子が、一定の外力と一定の電場のもとで行うブラウン運動を数値的にシミュレートし、粒子の平均初通過時間を求める。

## 確率微分方程式

### 導出

棒状粒子のシミュレーションを行うにあって、棒状粒子を二つの端点によって代表し（ダンベル型粒子モデル）、これら3点に外力と熱揺動力が作用するとして確率微分方程式を立てる。これから考えるようなブラウン運動では粘性抵抗が十分大きいので、系は過減衰状態であると仮定する。棒の両端の座標を $\bm{X}_{\pm}$ とすると、以下のようなランジュバン方程式が得られる。

$$
\begin{cases}
\gamma\bm{\dot{X}_+} &= \bm{F}^{int} + \bm{F}_+ + \bm{R}_+ \\
\gamma\bm{\dot{X}_-} &= -\bm{F}^{int} + \bm{F}_- + \bm{R}_-
\end{cases}
$$

ただし、$\gamma$ は粘性抵抗係数、$\bm{F}^{int}$ は棒の両端に働く内力、$\bm{F}_{\pm}$ は両端のと重心に働く外力、$\bm{R}_{\pm}$ は両端に働く熱揺動力である。これらの熱揺動力は平均がゼロであるガウス白色雑音であると仮定する。このランジュバン方程式を重心の並進運動と重心周りの回転運動に分解して、重心の座標を $\bm{Y}$ 、棒と $x$ 軸の成す角を $\Phi$ とすると、以下のような式が得られる。

$$
\left\{ 
\begin{array}{l}
\bm{\dot{Y}} = \frac{1}{2\gamma}(\bm{F}_+ + \bm{F}_- + \bm{R}_+ + \bm{R}_- )\\
\dot{\Phi} = \frac{1}{\gamma l}\bm{n}\times(\bm{F}_+ - \bm{F}_- + \bm{R}_+ - \bm{R}_-)
\end{array}
\right.
$$

ここで、$l$ は粒子の長さ、$\bm{n}$ は棒の向きを表す単位ベクトルである。また外積の計算では、2次元ベクトルを3次元空間に埋め込み、演算結果の $z$ 成分をとるものとする。

本シミュレーションにおいて、力 $\bm{F}_{\pm}$ は、常に $x$ 軸方向に一定の力 $\bm{f}$ 、壁からの反発力 $\bm{f}^{\text{rep}}_{\pm}$（[壁面の処理](#壁面の処理)で詳述）、および電気双極子が電場から受ける作用によって生まれるトルク $\tau$ による力の三つに分けられる。

$$
  \bm{F}_+ \coloneqq \bm{f} + \bm{f}^{\text{rep}}_+ + \frac{\tau}{\gamma}\bm{t},\quad \bm{F}_- \coloneqq \bm{f} + \bm{f}^{\text{rep}}_- - \frac{\tau}{\gamma}\bm{t}
$$

ただし、$\bm{t}$ は棒の向きに垂直な単位ベクトルである。トルク $\tau$ は電気双極子が電場から受ける作用によって生まれるものであるから、電場 $\bm{E}$ のもとでの電気双極子のエネルギー $U_{\bm{E}}$ を用いて、$\tau = -\frac{\partial}{\partial \Phi}U_{\bm{E}}$ と表される。今考えている粒子の永久双極子モーメントを $\bm{p}$ 、分極テンソルを $\hat{\alpha}$ とすると、電気双極子のエネルギーは

$$ U_{\bm{E}} = -\bm{p}\cdot\bm{E} - \frac{1}{2}\bm{E}^\top\hat{\alpha}\bm{E} $$

と表される。ここで、分極テンソル $\hat{\alpha}$ は棒の向きに沿った成分 $\alpha_{\parallel}$ と、棒の向きに垂直な成分 $\alpha_{\perp}$ を持つ対称なテンソルであると仮定する。すなわち、$\hat{\alpha} = \alpha_{\parallel}\bm{n}\bm{n}^\top + \alpha_{\perp}(\bm{I} - \bm{n}\bm{n}^\top)$ であるとすると、$U_{\bm{E}}$ は次のように表せる。

$$ U_{\bm{E}} = -\bm{p}\cdot\bm{E} - \frac{1}{2}\alpha_{\perp}|\bm{E}|^2 - \frac{1}{2}(\alpha_{\parallel} - \alpha_{\perp})(\bm{n}\cdot\bm{E})^2 $$

$\bm{n} = \left(\begin{array}{l}\cos\Phi \\ \sin\Phi\end{array}\right),\ \bm{E} = \left(\begin{array}{l}0 \\ E\end{array}\right)$ と設定すると

$$ U_{\bm{E}}(\Phi) = -pE\sin\Phi - \frac{E^2}{2}\alpha_{\perp} - \frac{E^2}{2}(\alpha_{\parallel} - \alpha_{\perp})\sin^2\Phi $$

より

$$ \tau(\Phi) = -\frac{\partial}{\partial \Phi}U_{\bm{E}}(\Phi) = pE\cos\Phi + \Delta\alpha E^2\sin \Phi\cos \Phi $$

ここで $\Delta\alpha \coloneqq \alpha_{\parallel} - \alpha_{\perp},\ \bm{p} = p\bm{n}$ である。以上から、本シミュレーションを行うための確率微分方程式は以下のようにまとめられる。

$$
\begin{aligned}
d\boldsymbol{Y} & =\frac{1}{\gamma} \boldsymbol{f} dt + \frac{1}{2\gamma}(\bm{f}^{\text{rep}}_+ + \bm{f}^{\text{rep}}_-)dt + \frac{1}{2\gamma}\left(d\boldsymbol{W}_{+} + d\boldsymbol{W}_{-}\right) \\
d\Phi & = \frac{2}{\gamma l^2}pE\cos\Phi(1+\frac{\Delta\alpha E}{p}\sin\Phi)dt + \frac{1}{\gamma l} \boldsymbol{n} \times\left(\bm{f}^{\text{rep}}_{+} - \bm{f}^{\text{rep}}_{-}\right)dt + \frac{1}{\gamma l} \boldsymbol{n} \times\left(d\boldsymbol{W}_{+} - d\boldsymbol{W}_{-}\right)
\end{aligned}
$$

$d\boldsymbol{W}_{j}(t)\ (j \in \{+,-\})$ は次の性質を満たす、互いに独立な Wiener 過程である。

$$
\mathbb{E}\left[d\boldsymbol{W}_{j}(t)\right] = \boldsymbol{0},\quad
\mathbb{E}\left[d\boldsymbol{W}_{j}(t) d\boldsymbol{W}_{j'}(t)^{\top}\right] = \frac{2\gamma}{\beta} \delta_{j,j'} \boldsymbol{I} dt 
$$

ただし、$\beta \coloneqq \frac{1}{k_B T}$ は熱力学的な逆温度である。

### 無次元化

シミュレーションの境界条件の周期の長さを $L$ とし、次のように無次元化したパラメータを定義する。

$$
\widetilde{\boldsymbol{Y}} \coloneqq \frac{1}{L}\boldsymbol{Y},\ 
\tilde{t} \coloneqq \frac{1}{\gamma\beta L^2}t,\ 
\tilde{l} \coloneqq \frac{1}{L}l,\ 
\widetilde{\bm{f}},\widetilde{\bm{f}}^{\text{rep}}_{\pm,0} \coloneqq \beta L\bm{f},\bm{f}^{\text{rep}}_{\pm,0},\ 
\widetilde{U} \coloneqq \beta U_{\bm{E}},\ 
\widetilde{d\boldsymbol{W}}(t) \coloneqq \frac{1}{\gamma L}d\boldsymbol{W}(t)
$$

すると、確率微分方程式は以下のように書き換えられる。

$$
\begin{aligned}
d\widetilde{\boldsymbol{Y}} &= \widetilde{\bm{f}} dt + \frac{1}{2}\left(\widetilde{\bm{f}}^{\text{rep}}_+ + \widetilde{\bm{f}}^{\text{rep}}_-\right)dt + \frac{1}{2}\left(\widetilde{d\boldsymbol{W}}_{+} + \widetilde{d\boldsymbol{W}}_{-}\right) \\
d\Phi &= \frac{1}{\tilde{l}}\left\{2\left(C_1\cos\Phi(1 + C_2\sin\Phi)\right)dt + \boldsymbol{n} \times\left(\widetilde{\bm{f}}^{\text{rep}}_{+} - \widetilde{\bm{f}}^{\text{rep}}_{-}\right)dt + \boldsymbol{n} \times\left(d\widetilde{\boldsymbol{W}}_{+} - d\widetilde{\boldsymbol{W}}_{-}\right)\right\}
\end{aligned}
$$

ただし、$C_1 \coloneqq \beta E(p/{\tilde{l}})$ 、$C_2 \coloneqq \Delta\alpha\frac{E}{p}$ である。ノイズ項についてさらに計算を進めると、次のように簡略化できる。

$$
\begin{aligned}
d\widetilde{\boldsymbol{Y}}_t &= \left\{ \widetilde{\bm{f}} + \frac{1}{2}(\widetilde{\bm{f}}^{\text{rep}}_+ + \widetilde{\bm{f}}^{\text{rep}}_-) \right\} dt + d\boldsymbol{W}_t \\
d\Phi_t &= \frac{1}{\tilde{l}}\left\{ 2 C_1\cos\Phi_t (1 + C_2\sin\Phi_t) + \boldsymbol{n} \times (\widetilde{\bm{f}}^{\text{rep}}_{+} - \widetilde{\bm{f}}^{\text{rep}}_{-}) \right\} dt + \frac{2}{\tilde{l}} dB_t
\end{aligned}
$$

ここで、$\boldsymbol{W}_t$ は2次元の標準ブラウン運動、$B_t$ は $\boldsymbol{W}_t$ と独立な1次元の標準ブラウン運動である。

$$
\begin{aligned}
  &\mathbb{E}\left[d\boldsymbol{W}_t\right] = \boldsymbol{0}
  &\mathbb{E}\left[d\boldsymbol{W}_t d\boldsymbol{W}_t^{\top}\right] = \boldsymbol{I} dt \\
  &\mathbb{E}\left[dB_t\right] = 0
  &\mathbb{E}\left[(dB_t)^2\right] = dt
\end{aligned}
$$

### 数値積分: 予測子・修正子法

上記のSDEは、単純な Euler–Maruyama 法ではなく予測子・修正子法によって数値積分される。壁の反発力はばね定数 $K$ が非常に大きいペナルティ法であり局所的な非線形性が強いため、ドリフト項を始点の状態だけで評価する Euler–Maruyama 法よりも、終点側の状態でドリフト項を評価し直す予測子・修正子法の方が数値的に安定する。

ドリフト項をまとめて

$$
\bm{b}(\widetilde{\boldsymbol{Y}}, \Phi) \coloneqq \begin{pmatrix} \widetilde{\bm{f}} + \frac{1}{2}(\widetilde{\bm{f}}^{\text{rep}}_+ + \widetilde{\bm{f}}^{\text{rep}}_-) \\[4pt] \frac{1}{\tilde{l}}\left\{ 2 C_1\cos\Phi (1 + C_2\sin\Phi) + \boldsymbol{n} \times (\widetilde{\bm{f}}^{\text{rep}}_{+} - \widetilde{\bm{f}}^{\text{rep}}_{-}) \right\} \end{pmatrix}
$$

と表すと、時刻 $t_n$ の状態 $(\widetilde{\boldsymbol{Y}}_n, \Phi_n)$ から時刻 $t_{n+1} = t_n + \Delta t$ の状態を求める1ステップは、以下の手順で計算される。まず、その時刻の1回だけ、揺動項の増分 $\Delta \boldsymbol{W}_n \sim \mathcal{N}(\boldsymbol{0}, \Delta t\, \boldsymbol{I})$ と $\Delta B_n \sim \mathcal{N}(0, \Delta t)$ を生成する。これらは予測子・修正子の両方で使い回し、ステップ内で再生成しない。

$$
\begin{aligned}
\text{予測子:}\quad & (\widetilde{\boldsymbol{Y}}^*, \Phi^*) = (\widetilde{\boldsymbol{Y}}_n, \Phi_n) + \bm{b}(\widetilde{\boldsymbol{Y}}_n, \Phi_n)\, \Delta t + \left(\Delta \boldsymbol{W}_n,\ \frac{2}{\tilde{l}}\Delta B_n\right) \\
\text{修正子:}\quad & (\widetilde{\boldsymbol{Y}}_{n+1}, \Phi_{n+1}) = (\widetilde{\boldsymbol{Y}}_n, \Phi_n) + \bm{b}(\widetilde{\boldsymbol{Y}}^*, \Phi^*)\, \Delta t + \left(\Delta \boldsymbol{W}_n,\ \frac{2}{\tilde{l}}\Delta B_n\right)
\end{aligned}
$$

すなわち、予測子でドリフト項を始点の状態 $(\widetilde{\boldsymbol{Y}}_n, \Phi_n)$ で評価して仮の終点 $(\widetilde{\boldsymbol{Y}}^*, \Phi^*)$ を求め(通常の Euler–Maruyama 法の1ステップに相当)、修正子ではそのドリフト項を仮の終点 $(\widetilde{\boldsymbol{Y}}^*, \Phi^*)$ で評価し直したうえで、改めて始点 $(\widetilde{\boldsymbol{Y}}_n, \Phi_n)$ に適用する。揺動項は状態に依存しないため、予測子・修正子のどちらでも同じ増分をそのまま用いる。

## 壁面の処理

粒子が運動できる領域は、上側の壁 $y = \omega(x)$ と下側の壁 $y = -\omega(x)$ に挟まれたチャネル状の空間であり、その形状は周期1の次の関数 $\omega$ によって定まる。

$$
\omega(x) \coloneqq \sin(2\pi x) + \frac{1}{4}\sin(4\pi x) + 1.12 = \sin(2\pi x)\left(\frac{1}{2}\cos(2\pi x) + 1\right) + 1.12
$$

### 反発力の計算

本シミュレーションでは、壁を完全に不透過な反射壁として扱うのではなく、壁面へのわずかなめり込みを許容したうえで、めり込みの深さに比例した力で押し戻す、いわゆるペナルティ法によって壁との相互作用を表現している。

棒の端点の一つを $\bm{X} = (p_x, p_y)$ とする。$-\omega(p_x) \le p_y \le \omega(p_x)$ が成り立てば端点はチャネルの内部にあり、反発力は生じないが、この不等式が破れる場合は端点は壁の外側にめり込んでいるので反発力を加える。端点がどちらの壁を越えたかを符号 $s \in \{+1,-1\}$（上壁なら $+1$、下壁なら $-1$）で表し、越えた側の壁を表す曲線を $y = \phi(x) \coloneqq s\,\omega(x)$ とおく。反発力の向きと大きさを決めるために、この曲線上で端点 $\bm{X}$ に最も近い点、すなわち $\bm{X}$ から曲線へ下ろした垂線の足 $(x^*, \phi(x^*))$ を求める。垂線の足は、点と曲線上の点との距離の2乗

$$
D(x) \coloneqq (x-p_x)^2 + (\phi(x)-p_y)^2
$$

を最小にする $x$ として特徴づけられ、次の停留条件（$D'(x) = 0$）を満たす。

$$
g(x) \coloneqq (x - p_x) + \phi'(x)\bigl(\phi(x) - p_y\bigr) = 0
$$

この非線形方程式を、コード内ではニュートン法によって数値的に解いている。

$$
x_{n+1} = x_n - \frac{g(x_n)}{g'(x_n)}, \qquad
g'(x) = 1 + \phi'(x)^2 + \phi''(x)\bigl(\phi(x) - p_y\bigr)
$$

初期値は $x_0 = p_x$ とし、5回反復して得られた値を解とする。

垂線の足 $(x^*, y^*) \coloneqq (x^*, \phi(x^*))$ が求まると、反発力はこの点に向かって端点を押し戻す、フックの法則に従うばね力として計算される。

$$
\bm{f}^{\text{rep}} \coloneqq K\bigl((x^*, y^*) - (p_x, p_y)\bigr)
$$

ここで $K$ はばね定数であり、（無次元化された量として）シミュレーション中では $K = 1.5\times10^{6}$ が用いられている。この力は端点から壁までの最短距離、すなわちめり込みの深さに比例した大きさを持ち、向きは常に壁面に垂直で、めり込みを解消する方向を向く。

棒状粒子は両端点 $\bm{X}_+, \bm{X}_-$ を持つため、以上の計算はそれぞれの端点について独立に行われる。得られた $\bm{f}^{\text{rep}}_+, \bm{f}^{\text{rep}}_-$ が、[確率微分方程式](#導出)の並進運動・回転運動の式にそのまま用いられる。
