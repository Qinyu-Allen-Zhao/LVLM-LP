def build_model(args):
    if args.model_name == "InstructBLIP":
        from .InstructBLIP import InstructBLIP
        model = InstructBLIP(args)
    elif args.model_name == "LLaVA-7B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "LLaVA-13B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "LLaMA_Adapter":
        from .LLaMA_Adapter import LLaMA_Adapter
        model = LLaMA_Adapter(args)
    elif args.model_name == "MMGPT":
        from .MMGPT import MMGPT
        model = MMGPT(args)
    elif args.model_name == "GPT4V":
        from .GPT4V import GPTClient
        model = GPTClient()
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG-Owl":
        from .mPLUG_Owl import mPLUG_Owl
        model = mPLUG_Owl(args)
    else:
        model = None
        
    return model
