def build_dataset(dataset_name, split, prompter):
    if dataset_name == "VizWiz":
        from .VizWiz import VizWizDataset
        dataset = VizWizDataset(prompter, split)
    elif dataset_name == "MAD":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split)
    elif dataset_name == "ImageNet":
        from .ImageNet import ImageNetDataset
        dataset = ImageNetDataset(split)
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split)
    elif dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split)
    elif dataset_name == "POPE":
        from .POPE import POPEDataset
        dataset = POPEDataset(split)
    else:
        from .base import BaseDataset
        dataset = BaseDataset(prompter)
        
    return dataset.get_data()
