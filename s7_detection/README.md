Для установки зависимостей создайте виртуальное окружение python 3.8.10 и в нём запустите команду *pip install -r requirements.txt*  

**s7_detection/notebooks/torchvision_finetuning_instance_segmentation.ipynb** - ноутбук, в котором иллюстрируется дообучения детекционной сети [Mask R-CNN](https://arxiv.org/abs/1703.06870) с использование библиотеки **torchvision**

**s7_detection/notebooks/segmentation.ipynb** - ноутбук, в котором иллюстрируется обучения сегментационной сети на базе [FPN](https://arxiv.org/abs/1612.03144) с использование библиотеки для сегментации на PyTorch [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

**s7_detection/notebooks/augmentations.ipynb** - ноутбук, в котором визуализируются аугметации, которые можно использовать при обучении детекционных сетей