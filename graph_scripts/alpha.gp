set term lua tikz
set output "alpha.tex"

unset key

set xlabel "$f$"             
set ylabel "$\\alpha$"

plot "../data/mono/alpha_150_01.dat" using 1:2 with points pt 7 

unset terminal
