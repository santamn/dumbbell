### アニメーション用 gnuplot スクリプト: 棒が動く軌跡とチャネル境界を描画する
###
### 入力データの形式: ステップ x座標  y座標 棒の角度
### 例:             123  0.0123 1.0456  -0.37

###############
### 設定項目 ###
###############

# 入出力ファイル
file = "data/di/trajectory_f1_seed1_omega_start.dat"
out  = "figures/di/trajectory_f1_seed1_omega_start.gif"

# アニメーション設定
frame_stride = 5000 # 何行ごとに1フレーム描くか：大きいほど枚数が少ない
frame_delay  = 10   # GIFの遅延（1/100秒単位）：大きいほど動画がスローになる

# チャネル境界: omega(x)
omega(x) = sin(2*pi*x) + 0.25*sin(4*pi*x) + 1.12

# 棒の長さ
L = 0.02

####################
### 以下は変更不要 ###
####################

x_margin = 0.05 # x方向の余白
nscan = 2000    # omega(x) の最大値を調べるための分割数

# 出力設定
set term gif animate optimize delay frame_delay size 900,700 # 出力GIFの設定
set output out                                               # 出力ファイル指定
set size ratio -1                                            # アスペクト比をデータの範囲に合わせる
set key off                                                  # 凡例非表示
set samples nscan                                            # 関数描画のサンプル数を指定
set tics out                                                 # 軸目盛を外側に出す

# 描画スタイル
set style line 1 lw 2 lc rgb "#000000"   # 境界
set style line 2 lw 1 lc rgb "#1f77b4"   # 軌跡
set style line 3 lw 2 lc rgb "#d62728"   # 棒

# xの範囲をデータから決める（余白つき）
stats file using 2 name "X" nooutput
if (!exists("X_records") || X_records < 1) {
    print sprintf("ERROR: failed to read data file: %s", file)
    exit
}
x_span = X_max - X_min
set xrange [X_min - x_margin*x_span : X_max + x_margin*x_span]

# yの範囲をデータと境界関数の両方が収まるように決める
stats file using 3 name "Y" nooutput
if (!exists("Y_records") || Y_records < 1) {
    print sprintf("ERROR: failed to read data file (y column): %s", file)
    exit
}
ymax_data = abs(Y_min)
if (abs(Y_max) > ymax_data) { ymax_data = abs(Y_max) }

ymax_omega = 0
do for [k=0:nscan] {
    xx = X_min + x_span*k/nscan
    yy = omega(xx)
    if (yy > ymax_omega) { ymax_omega = yy }
}
ymax = ymax_data
if (ymax_omega > ymax) { ymax = ymax_omega }
set yrange [-1.05*ymax : 1.05*ymax]

# アニメーションループ
#   - 境界: y=±omega(x)
#   - 軌跡: 1行目〜i行目まで
#   - 棒: i 行目の (x,y,theta) からベクトルとして描画
do for [i=1:X_records:frame_stride] {
    set title sprintf("trajectory (row=%d/%d)", i, X_records) # タイトルに行数表示

    plot \
        omega(x)   w l ls 1, \ # 上側境界
        -omega(x)  w l ls 1, \ # 下側境界
        file u 2:3 every ::1::i w l ls 2, \ # 軌跡
        file u ($2-0.5*L*cos($4)):($3-0.5*L*sin($4)):(L*cos($4)):(L*sin($4)) every ::i::i w vectors nohead ls 3 # 棒
}

set output
