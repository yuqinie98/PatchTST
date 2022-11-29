# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

for preLen in 96 192 336 720
do
# ETTm1
python -u run.py \
  --is_training 1 \
  --data_path ETTm1.csv \
  --task_id ETTm1 \
  --model FEDformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1  >../logs/LongForecasting/FEDformer_ETTm1_$pred_len.log

# ETTh1
python -u run.py \
  --is_training 1 \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model FEDformer \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LongForecasting/FEDformer_ETTh1_$pred_len.log

# ETTm2
python -u run.py \
  --is_training 1 \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model FEDformer \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LongForecasting/FEDformer_ETTm2_$pred_len.log

# ETTh2
python -u run.py \
  --is_training 1 \
  --data_path ETTh2.csv \
  --task_id ETTh2 \
  --model FEDformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LongForecasting/FEDformer_ETTh2_$pred_len.log

# electricity
python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_electricity_$pred_len.log

# exchange
python -u run.py \
 --is_training 1 \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model FEDformer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_exchange_rate_$pred_len.log

# traffic
python -u run.py \
 --is_training 1 \
 --data_path traffic.csv \
 --task_id traffic \
 --model FEDformer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3 >../logs/LongForecasting/FEDformer_traffic_$pred_len.log

# weather
python -u run.py \
 --is_training 1 \
 --data_path weather.csv \
 --task_id weather \
 --model FEDformer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_weather_$pred_len.log
done


for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --data_path national_illness.csv \
 --task_id ili \
 --model FEDformer \
 --data custom \
 --features S \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_ili_$pred_len.log
done

# cd ..