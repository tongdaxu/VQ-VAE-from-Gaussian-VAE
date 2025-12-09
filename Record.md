## FSQ 16xn
2025-06-30T03-46-47_-sd3unet_fsq_16_16x4

* imagenet 

PSNR: 26.3498 (±3.8711)
SSIM: 0.7569 (±0.1297)
MS-SSIM: 0.9338 (±0.0413)
LPIPS (AlexNet): 0.0749 (±0.0366)
FID: 1.1255

* coco

PSNR: 26.0103 (±3.4918)
SSIM: 0.7677 (±0.1092)
MS-SSIM: 0.9388 (±0.0325)
LPIPS (AlexNet): 0.0725 (±0.0290)
FID: 5.4516

2025-06-30T03-49-21_-sd3unet_fsq_32_16x8

* imagenet 

PSNR: 29.2974 (±4.1530)
SSIM: 0.8451 (±0.0956)
MS-SSIM: 0.9672 (±0.0234)
LPIPS (AlexNet): 0.0471 (±0.0302)
FID: 0.8741

* coco

PSNR: 29.0813 (±3.6920)
SSIM: 0.8558 (±0.0797)
MS-SSIM: 0.9705 (±0.0181)
LPIPS (AlexNet): 0.0434 (±0.0231)
FID: 4.0088

2025-06-30T03-52-44_-sd3unet_fsq_64_16x16

* imagenet

PSNR: 32.3808 (±4.2503)
SSIM: 0.9095 (±0.0605)
MS-SSIM: 0.9850 (±0.0113)
LPIPS (AlexNet): 0.0251 (±0.0196)
FID: 0.6365

* coco

PSNR: 32.3047 (±3.7462)
SSIM: 0.9171 (±0.0505)
MS-SSIM: 0.9866 (±0.0089)
LPIPS (AlexNet): 0.0223 (±0.0147)
FID: 2.7978

## BSQ 

2025-06-29T03-43-41_-sd3unet_bsq_16 epoch ?

* imagenet 

25.62 | 0.754 | 0.086 | 1.080 |

* coco

PSNR: 25.2961 (±3.1002)
SSIM: 0.7632 (±0.1033)
MS-SSIM: 0.9310 (±0.0314)
LPIPS (AlexNet): 0.0858 (±0.0291)
FID: 5.8038

2025-07-01T00-11-38_-sd3unet_bsq_32 epoch 25

imagenet 

PSNR: 27.8807 (±3.6231)
SSIM: 0.8362 (±0.0883)
MS-SSIM: 0.9581 (±0.0238)
LPIPS (AlexNet): 0.0597 (±0.0288)
FID: 0.7881

coco

PSNR: 27.5856 (±3.2523)
SSIM: 0.8441 (±0.0741)
MS-SSIM: 0.9609 (±0.0189)
LPIPS (AlexNet): 0.0576 (±0.0225)
FID: 4.4658

2025-07-01T00-13-03_-sd3unet_bsq_64 epoch 25

* imagenet

PSNR: 30.5086 (±3.6156)
SSIM: 0.9003 (±0.0546)
MS-SSIM: 0.9777 (±0.0127)
LPIPS (AlexNet): 0.0324 (±0.0172)
FID: 0.3458

* coco

PSNR: 30.3395 (±3.2139)
SSIM: 0.9065 (±0.0455)
MS-SSIM: 0.9795 (±0.0100)
LPIPS (AlexNet): 0.0311 (±0.0135)
FID: 2.6382

# ImageNet 128 generation 

* fsq 16 128

PSNR: 24.3045 (±3.4921)
SSIM: 0.7172 (±0.1307)
MS-SSIM: nan (±nan)
LPIPS (AlexNet): 0.0699 (±0.0289)
FID: 5.3188

100, cfg 4.0
(tensor(224.8893, device='cuda:0'), tensor(3.4019, device='cuda:0'))
tensor(7.3296, device='cuda:0')

* gaussian quant 128

PSNR: 25.5039 (±3.4448)
SSIM: 0.7767 (±0.1062)
MS-SSIM: nan (±nan)
LPIPS (AlexNet): 0.0579 (±0.0262)
FID: 2.3714

100, cfg 4.0
(tensor(230.7992, device='cuda:0'), tensor(4.1114, device='cuda:0'))
tensor(7.6766, device='cuda:0')

gq_128_log

* bsq 128

PSNR: 23.0831 (±2.9606)
SSIM: 0.7096 (±0.1168)
MS-SSIM: nan (±nan)
LPIPS (AlexNet): 0.0927 (±0.0330)
FID: 5.7553

100 cfg 4.0
(tensor(221.6450, device='cuda:0'), tensor(3.9233, device='cuda:0'))
tensor(7.8232, device='cuda:0')