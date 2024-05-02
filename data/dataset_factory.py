class DatasetFactory:
    _registry = {}

    @classmethod
    def register_dataset(cls, name, dataset_class):
        cls._registry[name] = dataset_class

    @classmethod
    def create_dataset(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} not registered.")
        return cls._registry[name](*args, **kwargs)
