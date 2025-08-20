# does not work
CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name simple_mlp_mcf_sm --cache_data_path ./data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --basic_model &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_mlp_mcf_sm_with_weight --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset transformer_mlp_mcf_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_joint_mlp --ip_history_window 15 --batch_size 256 --model_name simple_joint_mcf_sm --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset simple_joint_mcf_sm_4096 --basic_model &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_joint_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_joint_mcf_sm --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset transformer_joint_mcf_sm_4096 &
wait


CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_embedders --ip_history_window 15 --batch_size 256 --model_name mcf_sm_encoder --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset embedder_mcf_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name embedder_mlp_mcf_sm --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset simple_mlp_mcf_sm_4096 --encoder_name mcf_sm_encoder_cache &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -m jl.eval.eval_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_mlp_mcf_sm_with_weight --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset transformer_mlp_mcf_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -m jl.eval.eval_joint_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_joint_mcf_sm --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset transformer_joint_mcf_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -m jl.eval.eval_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_mlp_mcf_sm_with_weight --cache_data_path data/labeled_data/605.mcf_s-782B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_605.mcf_s-782B_sm.csv --dataset simple_mlp_mcf_sm_4096 --encoder_name mcf_sm_encoder_cache &
wait