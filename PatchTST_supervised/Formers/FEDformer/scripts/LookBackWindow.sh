# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LookBackWindow" ]; then
    mkdir ../logs/LookBackWindow
fi

for seqLen in 36 48 60 72 144 288
do
for pred_len in 24 576
do
python -u run.py \
  --is_training 1 \
  --root_path .../dataset/ \
  --data_path ETTm1.csv \
  --task_id ETTm1 \
  --model FEDformer \
  --data ETTm1 \
  --features M \
  --seq_len $seqLen \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LookBackWindow/FEDformer_ETTm2_$seqLen'_'$pred_len.log

python -u run.py \
  --is_training 1 \
  --root_path .../dataset/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model FEDformer \
  --data ETTm2 \
  --features M \
  --seq_len $seqLen \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LookBackWindow/FEDformer_ETTm2_$seqLen'_'$pred_len.log
done
done

for seqLen in 48 72 120 144 168 192 336 720
do
for pred_len in 24 720
do
# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path .../dataset/ \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model FEDformer \
  --data ETTh1 \
  --features M \
  --seq_len $seqLen \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LookBackWindow/FEDformer_ETTh1_$seqLen'_'$pred_len.log

# ETTh2
python -u run.py \
  --is_training 1 \
  --root_path .../dataset/ \
  --data_path ETTh2.csv \
  --task_id ETTh2 \
  --model FEDformer \
  --data ETTh2 \
  --features M \
  --seq_len $seqLen \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >../logs/LookBackWindow/FEDformer_ETTh2_$seqLen'_'$pred_len.log

## electricity
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_electricity_$seqLen'_'$pred_len.log

# exchange
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_exchange_rate_$seqLen'_'$pred_len.log

# traffic
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3 >../logs/LookBackWindow/FEDformer_traffic_$seqLen'_'$pred_len.log

# weather
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path weather.csv \
 --task_id weather \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_weather_$seqLen'_'$pred_len.log
done
done


for seqLen in 26 52 78 104 130 156 208
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 18 \
 --pred_len 24 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_ili_$seqLen'_'24.log

python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 18 \
 --pred_len 60 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_ili_$seqLen'_'60.log
done
# cd ..