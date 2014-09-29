import multiprocessing
import pandas as pd

__author__ = 'johannes'


class Record(object):
    # record = pd.DataFrame()
    records = {}

    def __init__(self, columns):
        self.columns = columns

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            data = f(*args, **kwargs)

            if data is None:
                return data

            _self = args[0]
            columns = list(self.columns)

            if hasattr(_self, "index"):
                for i, _ in enumerate(self.columns):
                    columns[i] += "_" + str(_self.index)

            rec = pd.DataFrame([data], index=_self.world.get_index(),
                               columns=columns)

            pid = multiprocessing.current_process().pid
            old_record = Record.records.get(pid, pd.DataFrame())
            Record.records[pid] = old_record.combine_first(rec)

            return data

        return wrapped_f
