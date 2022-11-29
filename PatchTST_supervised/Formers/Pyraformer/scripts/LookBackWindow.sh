# cd Pyraformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LookBackWindow" ]; then
    mkdir ../logs/LookBackWindow
fi

for seqlen in 24 48 72 96 120 144 168 336 504 672 720
    do 
    for pred_len in 720
    do
    python long_range_main.py -window_size [2,2,2] -data_path electricity.csv -data electricity -input_size $seqlen -predict_step $pred_len -n_head 6 -lr 0.00001 -d_model 256 >../logs/LookBackWindow/Pyraformer_electricity_$seqlen'_'$pred_len.log
    python long_range_main.py -window_size [2,2,2] -data_path exchange_rate.csv -data exchange -input_size $seqlen -predict_step $pred_len -n_head 6 -lr 0.00001 -d_model 256  >../logs/LookBackWindow/Pyraformer_exchange_rate_$seqlen'_'$pred_len.log
    python long_range_main.py -window_size [2,2,2] -data_path traffic.csv -data traffic -input_size $seqlen -predict_step $pred_len  -n_head 6 -lr 0.00001 -d_model 256 >../logs/LookBackWindow/Pyraformer_traffic_$seqlen'_'$pred_len.log
    python long_range_main.py -window_size [2,2,2] -data_path weather.csv -data weather -input_size $seqlen -predict_step $pred_len -n_head 6 -lr 0.00001 -d_model 256 >../logs/LookBackWindow/Pyraformer_weather_$seqlen'_'$pred_len.log
    python long_range_main.py -window_size [2,2,2] -data ETTh1 -input_size $seqlen -predict_step $pred_len -n_head 6 >../logs/LookBackWindow/Pyraformer_ETTh1_$seqlen'_'$pred_len.log
    python long_range_main.py -window_size [2,2,2] -data ETTh2 -data_path ETTh2.csv -input_size $seqlen -predict_step $pred_len -n_head 6 >../logs/LookBackWindow/Pyraformer_ETTh2_$seqlen'_'$pred_len.log
done
done

for seqlen in 26 52 78 104 130 156 208
    do 
    for pred_len in 24 60
    do
    python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili -input_size $seqlen -predict_step $pred_len -n_head 6 -lr 0.00001 -d_model 256 >../logs/LookBackWindow/Pyraformer_ili_$seqlen'_'$pred_len.log
done
done

for seqlen in 24 36 48 60 72 144 288
    do 
    for pred_len in 24 576
    do
    python long_range_main.py  -window_size [2,2,2] -data ETTm1 -data_path ETTm1.csv -input_size $seqlen -predict_step $pred_len -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LookBackWindow/Pyraformer_ETTm1_$seqlen'_'$pred_len.log
    python long_range_main.py  -window_size [2,2,2] -data ETTm2 -data_path ETTm2.csv -input_size $seqlen -predict_step $pred_len -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LookBackWindow/Pyraformer_ETTm2_$seqlen'_'$pred_len.log
done
done
# cd ..
