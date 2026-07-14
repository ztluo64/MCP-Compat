import torch
from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset


class COinCOTriViewDatasetV2(COinCOTriViewDataset):
    """
    Same as COinCOTriViewDataset, but additionally returns replacement_label.
    """
    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if hasattr(self, "df"):
            row = self.df.iloc[idx]
        elif hasattr(self, "data"):
            row = self.data.iloc[idx]
        else:
            raise AttributeError("Dataset has neither self.df nor self.data.")

        if "replacement_label" not in row:
            raise KeyError("replacement_label not found in csv. Run add_replacement_label.py first.")

        item["replacement_label"] = torch.tensor(int(row["replacement_label"]), dtype=torch.long)
        return item
