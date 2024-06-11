from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.relabelers.relabeler import Relabeler

class RelabelerConstant(Relabeler):
    
    def __init__(self, new_label=0) -> None:
        super().__init__()

        self.new_label = new_label

    def fit(self, labels):
        return self
    
    def transform(self, labels):
        n_labels = len(labels)

        new_labels = [self.new_label for _ in range(n_labels)]

        return new_labels