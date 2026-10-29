"""Conservative partitioning by target hour and input-hour start."""
import pandas as pd

def window(frame,horizon,start,end,issue_not_before):
    """Input-hour start >= issue_not_before; target-hour start in [start,end)."""
    if horizon<=0: raise ValueError('horizon must be positive')
    target=frame.index+pd.Timedelta(hours=horizon)
    return frame[(frame.index>=issue_not_before)&(target>=start)&(target<end)]
