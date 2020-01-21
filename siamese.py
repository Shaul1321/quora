import pickle
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import loss
import scipy
from scipy import linalg


class Dataset(torch.utils.data.Dataset):
    
    """Simple torch dataset class"""
    def __init__(self, data, device):

        self.data = data
        self.device = device

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        with torch.no_grad():
             
            vec1_np, vec2_np, str1, str2, _ = self.data[index]
            
            vec1, vec2, str1, str2, pair_id = self.data[index]            
            vec1, vec2 = torch.from_numpy(vec1_np).float(), torch.from_numpy(vec2_np).float()
            vec1 = vec1.to(self.device)
            vec2 = vec2.to(self.device)
            
            return (vec1, vec2, str1, str2, pair_id)
            
            
            
class Siamese(pl.LightningModule):

    def __init__(self, X_train, X_dev, dim, batch_size, dropout_rate, device = "cuda"):
        super(Siamese, self).__init__()
        self.l = torch.nn.Linear(1024, dim)
        
        self.train_data = Dataset(X_train, device)
        self.dev_data = Dataset(X_dev, device)
        self.train_gen = torch.utils.data.DataLoader(self.train_data, batch_size = batch_size, drop_last = False, shuffle=True)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_data, batch_size = batch_size, drop_last = False, shuffle=True)
        self.loss_fn = loss.BatchHardTripletLoss(final = "softmax", device = device)
        self.dropout = torch.nn.Dropout(p = dropout_rate)
        self.acc = None
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay = 1e-6)
        
    def forward(self, x1, x2):

          h1 = self.l(x1)
          h2 = self.l(x2)
         
          return h1, h2
 
    def train_network(self, num_epochs):
    
      trainer = Trainer(max_nb_epochs = num_epochs, min_nb_epochs = num_epochs, show_progress_bar = False)
      trainer.fit(self)

      return self.acc   
      
    def get_weights(self):
    
        return self.l.weight.data.detach().cpu().numpy()
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x1, x2, str1, str2, ids = batch
        h1, h2 = self.forward(self.dropout(x1), self.dropout(x2))
        loss_val =  self.loss_fn(h1, h2, str1, str2, ids, index=0, evaluation = False)
        
        return {'loss': loss_val[0]}
        

    def validation_step(self, batch, batch_nb):
    
        # OPTIONAL
        x1, x2, str1, str2, ids = batch
        h1, h2 = self.forward(x1, x2)
        loss_val =  self.loss_fn(h1, h2, str1, str2, ids, index=batch_nb, evaluation = True)
        return {'val_loss': loss_val[0]}

    def validation_end(self, outputs):
        # OPTIONAL    
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print("Loss is {}".format(avg_loss))
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), weight_decay = 1e-4)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_gen

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return self.dev_gen
