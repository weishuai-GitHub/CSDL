CUDA_VISIBLE_DEVICES=0 python train_mlp.py \
--dataset_name=cifar10 \
--alpha=2.0  \
--mlp_out_dim=500 \
--epochs=150 \
--batch_size=200 \
--constrative_subspace_weight=1.0 \
--temperature=0.1
CUDA_VISIBLE_DEVICES=0 python train_mlp.py \
--dataset_name=cifar10 \
--alpha=2.0  \
--mlp_out_dim=1000 \
--epochs=150 \
--batch_size=200 \
--constrative_subspace_weight=1.0 \
--temperature=0.1
CUDA_VISIBLE_DEVICES=0 python train_mlp.py \
--dataset_name=cifar10 \
--alpha=2.0  \
--mlp_out_dim=3000 \
--epochs=150 \
--batch_size=200 \
--constrative_subspace_weight=1.0 \
--temperature=0.1
CUDA_VISIBLE_DEVICES=4 python train_mlp.py \
--dataset_name=cifar10 \
--alpha=2.0  \
--mlp_out_dim=4000 \
--epochs=150 \
--batch_size=200 \
--constrative_subspace_weight=1.0 \
--temperature=0.1