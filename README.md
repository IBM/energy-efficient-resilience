# sample run command - 
# Training AI models to jointly optimize energy efficiency and resilience
# To run an 8-bit model with error injection at the rate of 0.01
python3 zs_main.py resnet18 eval cifar10 -cp checkpoint.pth -p 8 -err 1 -ber 0.01
=======
# energy-efficient-resilience
