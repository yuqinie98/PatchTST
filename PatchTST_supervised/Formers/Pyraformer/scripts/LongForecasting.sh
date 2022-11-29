# cd Pyraformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# ETTh1
for pred_len in 96 192 336
do
python long_range_main.py -data ETTh1 -input_size 96 -predict_step $pred_len -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
done
python long_range_main.py -data ETTh1 -input_size 96 -predict_step 720 -inner_size 5 -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_720.log

# # ETTh2
for pred_len in 96 192 336
do
python long_range_main.py -data ETTh2 -input_size 96 -data_path ETTh2.csv -predict_step $pred_len -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh2_$pred_len.log
done
python long_range_main.py -data ETTh2 -input_size 96 -data_path ETTh2.csv -predict_step 720 -inner_size 5 -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh2_720.log

# ETTm1
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 96 \
 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm1_96.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 192 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm1_192.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 336 \ 
 -inner_size 5 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm1_336.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 720 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm1_720.log

# ETTm2
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 96 \
 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm2_96.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 192 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm2_192.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 336 \
 -inner_size 5 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm2_336.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 720 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >../logs/LongForecasting/Pyraformer_ETTm2_720.log

# ili
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 24 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_ili_24.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 36 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_ili_36.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 48 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_ili_48.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 60 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_ili_60.log

# Electricity
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_electricity_96.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_electricity_192.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_electricity_336.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_electricity_720.log

# Exchange
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_96.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_192.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_336.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_720.log

# Traffic
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_traffic_96.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_traffic_192.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_traffic_336.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 720  -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_traffic_720.log

# Weather
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_weather_96.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_weather_192.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_weather_336.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_weather_720.log

# cd ..
