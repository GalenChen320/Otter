from .sweci import show_sweci_summary
from .evalplus import show_evalplus_summary


from otter.config.setting import get_settings

def summarize():
    dataset = get_settings().dataset_name
    match dataset:
        case "sweci": 
            show_sweci_summary()
        case "evalplus" | "mbppplus": 
            show_evalplus_summary()
        case _: 
            raise NotImplementedError(dataset)


__all__ = [
    "summarize"
]