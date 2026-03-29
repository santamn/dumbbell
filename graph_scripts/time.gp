set term lua tikz
set output "time.tex"

set key top left offset 3, -2
set logscale x

set xlabel "$|f|$"
set ylabel "First Passage Time"

plot "../data/len0.08_K1.5e6/time.dat" using 1:2 with points pt 7  title "Forward", \
      "../data/len0.08_K1.5e6/time.dat" using 1:3 with points pt 12 title "Reverse"

unset terminal
