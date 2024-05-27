# ConvHuberMC-Net

*ConvHuberMC-Net* takes the corrupted matrix, perhaps some sensor data, as input, along with two smaller matrices whose product has the same dimensions as the corrupted matrix. The smaller matrices are then processed via the Huber regression algorithm in each layer of the neural network, potentially being refined in a way that makes their product, the recovered matrix, reasonably similar to the pristine version of the corrupted matrix.

- [Code](https://github.com/Talha-Nehal-Undegrad-Study/ConvHuberMC-Net/blob/main/convhubermc.ipynb)
- [Paper](https://github.com/Talha-Nehal-Undegrad-Study/ConvHuberMC-Net/blob/main/Literature/Sproj%20Report/SPROJ_Report.pdf)
