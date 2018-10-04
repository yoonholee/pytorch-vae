hsizes=(80 160 320 640)
zsizes=(10 20 40 80)
for h in "${hsizes[@]}"
do
for z in "${zsizes[@]}"
do
    dfc start main.py --h_dim=$h --z_dim=$z & sleep 5;
done
done

