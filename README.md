# cityscape-segmentation-Unet
Cityscape segmentation Unet architecture using pytorch

# IndonesiaForAI-Project-3-SelF-Driving-Car
Segmentasi citra perkotaan dengan menggunakan U-Net melibatkan penerapan arsitektur jaringan saraf tiruan berbentuk U yang dirancang untuk klasifikasi piksel yang efektif dalam adegan perkotaan. Model ini terdiri dari jalur kontraksi (encoder) untuk menangkap konteks dan mengekstrak fitur, serta jalur ekspansi (decoder) untuk mengembalikan dimensi spasial dan menjaga detail beresolusi tinggi melalui koneksi skip. Selama pelatihan, U-Net dioptimalkan menggunakan kerugian cross-entropy pada dataset berlabel, memungkinkannya belajar untuk mengklasifikasikan piksel dengan akurat ke dalam kelas-kelas tertentu seperti jalan, bangunan, dan kendaraan. Setelah dilatih, model dapat diterapkan pada citra perkotaan baru, memberikan masker segmentasi yang menyoroti berbagai objek dan wilayah, menjadikannya alat berharga untuk aplikasi seperti navigasi otonom, perencanaan perkotaan, dan analisis lingkungan.

![image](https://github.com/fahernkhan/cityscape-segmentation-Unet/assets/128980804/9ab20fc8-7a1e-4719-a5d2-6264b689c223)


## Import Library

```sh
import torch
import torchvision
from glob import glob
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
```


## Using From Cityscapes Image Pairs Dataset

![image](https://github.com/fahernkhan/cityscape-segmentation-Unet/assets/128980804/37d786d2-4754-41fe-ade6-f020beb22a3c)
