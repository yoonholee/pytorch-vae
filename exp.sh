run="pipenv run python main.py"

$run --gpu=0 &
$run --gpu=1 --importance_num=64 &
$run --gpu=2 --importance_num=8 --mean_num=8 &
$run --gpu=3 --no_iwae_lr &
$run --gpu=4 --z=100 &
wait
