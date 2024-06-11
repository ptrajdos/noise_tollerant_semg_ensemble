
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_dummy import SetCreatorDummy


class SetCreatorTransformerWrapper:
    def __init__(self, set_creator:SetCreator=SetCreatorDummy()) -> None:
        self.set_creator = set_creator

    def fit(self, X, y=None):
        self.set_creator.fit(X,y)
        return self
    
    def transform(self, X):
        Xt, yt, t = self.set_creator.transform(X)

        return Xt
    
    def fit_transform(self, X, y=None):

        Xt, yt, t = self.set_creator.fit_transform(X, y)
        return Xt