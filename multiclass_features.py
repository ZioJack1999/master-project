"""Past-window filter-bank covariance features and small neural classifiers.

Spatial/log-power structure is inspired by shallow EEG ConvNets (Schirrmeister
et al. 2017); the covariance branch uses a train-referenced matrix logarithm.
No test data fit scales, covariance references, or network parameters.
"""
import numpy as np
import torch
from torch import nn

BANDS=[(8.,12.),(12.,16.),(16.,22.),(22.,30.)]
MOTOR10=['F3','Fz','F4','C3','Cz','C4','P3','Pz','P4','POz']

def band_covariances(X,fs=256,batch_size=128):
    X=np.asarray(X);n=X.shape[-1];frequencies=np.fft.rfftfreq(n,1/fs)
    taper=np.hanning(n);scale=2./(n*np.sum(taper**2));out=[]
    for a in range(0,len(X),batch_size):
        x=X[a:a+batch_size];fft=np.fft.rfft((x-x.mean(axis=-1,keepdims=True))*taper,axis=-1)
        bands=[]
        for low,high in BANDS:
            spec=fft[...,(frequencies>=low)&(frequencies<high)]
            bands.append((np.einsum('bcf,bdf->bcd',spec.real,spec.real)+np.einsum('bcf,bdf->bcd',spec.imag,spec.imag))*scale)
        out.append(np.stack(bands,axis=1))
    return np.concatenate(out).astype(np.float32)

def spd_function(matrices,function):
    values,vectors=np.linalg.eigh((matrices+matrices.swapaxes(-1,-2))*.5)
    values=np.maximum(values,1e-8)
    return (vectors*function(values)[...,None,:])@vectors.swapaxes(-1,-2)

def shrink_covariances(covariances,amount):
    n=covariances.shape[-1];trace=np.trace(covariances,axis1=-2,axis2=-1)/n
    return (1-amount)*covariances+amount*trace[...,None,None]*np.eye(n)

class FeaturePreprocessor:
    def __init__(self,parameters):self.parameters=parameters
    @classmethod
    def fit(cls,covariances,weights,family,shrinkage=.1):
        weights=np.asarray(weights)/np.sum(weights)
        avg=np.einsum('b,bkij->kij',weights,covariances)
        scale=np.sqrt(np.maximum(np.diagonal(avg.sum(axis=0)),1e-16))
        p=dict(family=family,channel_scale=scale,shrinkage=shrinkage)
        obj=cls(p);normalized=obj.normalized(covariances)
        if family=='log_covariance':
            reference=np.einsum('b,bkij->kij',weights,shrink_covariances(normalized,shrinkage))
            p['reference_inverse_sqrt']=spd_function(reference,lambda x:x**-.5)
        features=obj.unscaled_features(normalized)
        mean=np.einsum('b,bf->f',weights,features)
        std=np.sqrt(np.einsum('b,bf->f',weights,(features-mean)**2))
        p.update(feature_mean=mean,feature_std=np.maximum(std,.05))
        return obj
    def normalized(self,covariances):
        scale=np.asarray(self.parameters['channel_scale'])
        return (covariances/scale[None,None,:,None]/scale[None,None,None,:]).astype(np.float32)
    def unscaled_features(self,normalized):
        if self.parameters['family']=='filterbank_power':
            return np.log(np.maximum(np.diagonal(normalized,axis1=-2,axis2=-1),1e-6)).reshape(len(normalized),-1)
        ref=np.asarray(self.parameters['reference_inverse_sqrt'])
        cov=shrink_covariances(normalized,self.parameters['shrinkage'])
        tangent=spd_function(ref[None]@cov@ref[None],np.log)
        i,j=np.triu_indices(cov.shape[-1]);values=tangent[...,i,j]
        values*=np.where(i==j,1.,np.sqrt(2.))
        return values.reshape(len(cov),-1).astype(np.float32)
    def transform(self,covariances):
        normalized=self.normalized(covariances)
        if self.parameters['family']=='filterbank_power':return normalized
        p=self.parameters
        return ((self.unscaled_features(normalized)-p['feature_mean'])/p['feature_std']).astype(np.float32)

class FeatureNetwork(nn.Module):
    def __init__(self,family,n_channels,n_classes,preprocessing,hidden=32,dropout=.25):
        super().__init__();self.family=family
        if family=='filterbank_power':
            initial=torch.eye(n_channels)[None].repeat(len(BANDS),1,1)
            self.spatial=nn.Parameter(initial+.01*torch.randn_like(initial))
            n_features=len(BANDS)*n_channels
            self.register_buffer('feature_mean',torch.as_tensor(preprocessing['feature_mean'],dtype=torch.float32))
            self.register_buffer('feature_std',torch.as_tensor(preprocessing['feature_std'],dtype=torch.float32))
        else:n_features=len(BANDS)*n_channels*(n_channels+1)//2
        if hidden == 0:
            self.head = nn.Linear(n_features, n_classes)
        else:
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(n_features,hidden),nn.ELU(),nn.Dropout(dropout),nn.Linear(hidden,n_classes))
    def forward(self,x):
        if self.family=='filterbank_power':
            weights=self.spatial/(self.spatial.norm(dim=-1,keepdim=True)+1e-8)
            power=torch.einsum('kfc,bkcd,kfd->bkf',weights,x,weights)
            x=(power.clamp_min(1e-6).log().flatten(1)-self.feature_mean)/self.feature_std
        return self.head(x)

class FeatureEnsembleRunner:
    def __init__(self,checkpoint,device='cpu'):
        self.device = torch.device(device)
        self.checkpoint=checkpoint;self.preprocessor=FeaturePreprocessor(checkpoint['preprocessing'])
        self.members=[];self.logit_bias=np.asarray(checkpoint.get('post_logit_bias',np.zeros(checkpoint['n_classes'])),np.float32)
        if self.logit_bias.shape!=(checkpoint['n_classes'],) or not np.isfinite(self.logit_bias).all():raise ValueError('Invalid neural class-bias calibration.')
        for state in checkpoint['member_states']:
            model=FeatureNetwork(checkpoint['family'],checkpoint['n_channels'],checkpoint['n_classes'],checkpoint['preprocessing'],checkpoint['hidden_units'],checkpoint['dropout'])
            model.load_state_dict(state)
            self.members.append(model.to(self.device).eval())
    def from_covariances(self,covariances,batch_size=256):
        features=self.preprocessor.transform(covariances);all_probabilities=[]
        with torch.inference_mode():
            for a in range(0,len(features),batch_size):
                x = torch.from_numpy(features[a:a+batch_size]).to(self.device)
                probabilities = torch.stack([torch.softmax(model(x), dim=1) for model in self.members])
                all_probabilities.append(probabilities.mean(dim=0).cpu().numpy())
        p=np.concatenate(all_probabilities)
        logits=np.log(np.maximum(p,1e-12))+self.logit_bias[None]
        logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits)
        return p/p.sum(axis=1,keepdims=True)
    def predict(self,X):return self.from_covariances(band_covariances(X))
