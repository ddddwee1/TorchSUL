import sys 
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           ProgressColumn, TextColumn, TimeRemainingColumn)
from rich.text import Text


# progressbar utils 
class SpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        else:
            return Text(f"{speed:.2f} it/s", style="progress.data.speed")

def progress_bar(width: int=40, disable: bool=False) -> Progress:
    # is_debug_mode = False
    # gettrace = getattr(sys, 'gettrace', None)
    # if gettrace is not None:
    #     if gettrace() is not None:
    #         is_debug_mode = True

    prog = Progress(
        TextColumn('[progress.description]{task.description}'), 
        BarColumn(finished_style='green', bar_width=width), 
        MofNCompleteColumn(), 
        TimeRemainingColumn(elapsed_when_finished=True), 
        SpeedColumn(),
        disable = disable,
        )
    return prog
