"""The baseline columnwise preprocessing restricted to declared predictor columns."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class SubsetPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,sex_index=None):self.sex_index=sex_index
    def fit(self,X,y=None):
        x=np.asarray(X,dtype=float)
        if np.isnan(x).all(axis=0).any():raise ValueError('Entire fitting predictor missing')
        self.lower_=np.nanquantile(x,.01,axis=0);self.upper_=np.nanquantile(x,.99,axis=0)
        if self.sex_index is not None:self.lower_[self.sex_index],self.upper_[self.sex_index]=0,1
        z=np.clip(x,self.lower_,self.upper_);self.fill_=np.nanmedian(z,axis=0)
        if self.sex_index is not None:
            values,counts=np.unique(x[np.isfinite(x[:,self.sex_index]),self.sex_index],return_counts=True)
            self.fill_[self.sex_index]=values[np.argmax(counts)]
        z=np.where(np.isnan(z),self.fill_,z);self.mean_,self.scale_=z.mean(axis=0),z.std(axis=0)
        self.scale_[self.scale_==0]=1
        return self
    def transform(self,X):
        z=np.clip(np.asarray(X,dtype=float),self.lower_,self.upper_)
        return (np.where(np.isnan(z),self.fill_,z)-self.mean_)/self.scale_
