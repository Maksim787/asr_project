class BaseMetric:
    def __init__(self, name=None, only_val: bool = False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.only_val = only_val
        self.kwargs = kwargs

    def __call__(self, **batch):
        raise NotImplementedError()
