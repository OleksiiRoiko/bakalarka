from .cifar10_dataset import (CIFAR10WithBackground,
                              CIFAR10WithDistributedSoftBackgroundLabels,
                              CIFAR10WithBackgroundSoftLabel,
                              DefaultCIFAR10,
                              CIFAR10WithClassSpecificBackground,
                              CIFAR10WithClassSpecificBackgroundSoftLabel,
                              CIFAR10WithDistributedClassSpecificSoftBackgroundLabels,
                              SegmentedCIFAR10WithObject,
                              CIFAR10Utils)
from .dataset_factory import DatasetFactory

DatasetFactory.register_dataset("DefaultCIFAR10", DefaultCIFAR10)
DatasetFactory.register_dataset("CIFAR10WithBackground", CIFAR10WithBackground)
DatasetFactory.register_dataset("CIFAR10WithBackgroundSoftLabel", CIFAR10WithBackgroundSoftLabel)
DatasetFactory.register_dataset("CIFAR10WithDistributedSoftBackgroundLabels", CIFAR10WithDistributedSoftBackgroundLabels)
DatasetFactory.register_dataset("CIFAR10WithClassSpecificBackground", CIFAR10WithClassSpecificBackground)
DatasetFactory.register_dataset("CIFAR10WithClassSpecificBackgroundSoftLabel", CIFAR10WithClassSpecificBackgroundSoftLabel)
DatasetFactory.register_dataset("CIFAR10WithDistributedClassSpecificSoftBackgroundLabels", CIFAR10WithDistributedClassSpecificSoftBackgroundLabels)
DatasetFactory.register_dataset("SegmentedCIFAR10WithObject", SegmentedCIFAR10WithObject)
