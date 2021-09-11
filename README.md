# Edge-Enhancement

Implementions of "Edge Enhancement Improves Adversarial Robustness in Image Classification" and other advserarial training method, including Standard Training(ST), Advserarial Training(AT), ALP, TRADES and Avmixup

## Usage
For MNIST and Tiny ImageNet, we use DataParallel(DP). 
For ImageNet, we use DistributedDataParallel (DDP).

### MNIST
```python
python experiments_mnist.py --data "MNIST Data" --config "Path of a config in configs_mnist"
```

### Tiny ImageNet
```python
python experiments_tinyimagenet.py --data "Tiny ImageNet Data" --config "Path of a config in configs_tinyimagenet"
```

### ImageNet
```python
python -m torch.distributed.launch --nproc_per_node=int(the numbers of GPUs) --master_port=12345 experiments_imagenet.py --data "ImageNet Data" --config "Path of a config in configs_imagenet"
```