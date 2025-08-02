CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name simple_mlp_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset simple_mlp_xalan_sm_4096 --basic_model &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_mlp_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset transformer_mlp_xalan_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_joint_mlp --ip_history_window 15 --batch_size 256 --model_name simple_joint_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset simple_joint_xalan_sm_4096 --basic_model &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_joint_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_joint_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset transformer_joint_xalan_sm_4096 &
wait


CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_embedders --ip_history_window 15 --batch_size 256 --model_name xalan_sm_encoder --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset embedder_xalan_sm_4096 &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_mlp --ip_history_window 15 --batch_size 256 --model_name embedder_mlp_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset simple_mlp_xalan_sm_4096 --encoder_name xalan_sm_encoder_cache &
wait

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.eval.eval_joint_mlp --ip_history_window 15 --batch_size 256 --model_name transformer_joint_xalan_sm --cache_data_path data/labeled_data/623.xalancbmk_s-10B_sm_cs_4096_labeled.csv -p data/collector_output/prefetches_623.xalancbmk_s-10B_sm.csv --dataset transformer_joint_xalan_sm_4096 &
