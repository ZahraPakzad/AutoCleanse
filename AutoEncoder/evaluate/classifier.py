import os
import io
import torch
import torch.nn as nn
from AutoEncoder.utils import *
from collections import Counter
from exasol.bucketfs import Service
from tqdm import tqdm
from AutoEncoder.loss_model import loss_CEMSE
from AutoEncoder.bucketfs_client import BucketFS_client
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

class ClassifierDummy:
    def __init__(self, strategy='most_frequent', n_splits=10, n_repeats=3, random_state=42):
        self.model = DummyClassifier(strategy=strategy)
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    def compute_scores(self, X, y):
        scores = cross_val_score(self.model, X, y, scoring='accuracy', cv=self.cv, n_jobs=-1)
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))        

class BasicNet(nn.Module):
    
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = 0
        
        self.lin1 = torch.nn.Linear(self.num_features,  150)                
        self.lin4 = torch.nn.Linear(150, 150)     
        self.lin10 = torch.nn.Linear(150, self.num_classes)
        
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, xin):
        self.layers = 0
        
        x = F.relu(self.lin1(xin))
        self.layers += 1
        
        #x = F.relu(self.lin2(x))
        #self.layers += 1
        for y in range(8):
          x = F.relu(self.lin4(x)) 
          self.layers += 1
           
        x = self.dropout(x)
        
        x = F.relu(self.lin10(x)) 
        self.layers += 1
        return x

    def train(self, model, train_loader, optimizer, epoch):
        model.train()
        
        for inputs, target in train_loader:
        
            inputs, target = inputs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            # Backprop
            loss.backward()
            optimizer.step()

    def test(self, model, test_loader):
        model.eval()
        
        test_loss = 0
        correct = 0
        test_size = 0
        
        with torch.no_grad():
        
            for inputs, target in test_loader:
                
                inputs, target = inputs.to(device), target.to(device)
                
                output = model(inputs)
                test_size += len(inputs)
                test_loss += test_loss_fn(output, target).item() 
                pred = output.max(1, keepdim=True)[1] 
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        accuracy = correct / test_size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_size,
            100. * accuracy))
        
        return test_loss, accuracy
    

class ClsNNBase(nn.Module):
    def __init__(self, layers, l1_strength, l2_strength, batch_norm, dropout):
        super(ClsNNBase, self).__init__()
        self.num_layers = len(layers)
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

        hidden_layers = []
        for i in range(self.num_layers - 1):
            hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if batch_norm == True:
                hidden_layers.append(nn.BatchNorm1d(layers[i + 1]))
            hidden_layers.append(nn.ReLU())
            hidden_layers[-1].register_forward_hook(self.add_regularization_hook) 
            if dropout is not None:
                for drop_layer, drop_chance in dropout:
                    if i == drop_layer:
                        hidden_layers.append(nn.Dropout(drop_chance))
        hidden_layers.append(nn.Sigmoid())                    
        self.network = nn.Sequential(*hidden_layers)

    def add_regularization_hook(self, module, input, output):
        l1_reg = self.l1_strength * F.l1_loss(output, torch.zeros_like(output))
        l2_reg = self.l2_strength * F.mse_loss(output, torch.zeros_like(output))
        module.register_forward_hook(None)  
        module._forward_hooks.clear()
        return output + l1_reg + l2_reg 

    def forward(self, x):
        x = self.network(x)
        return x

    @classmethod
    def build_model(cls, layers, dropout, batch_norm, learning_rate=1e-3, weight_decay=0, l1_strength=0, l2_strength=0, load_method=None, weight_path=None):
        """
        @brief Build autoencoder encoder decoder and optimizer.
        @param layers: A list specifying the number of layers and their respective size
        @param dropout: A list of tuple specifying dropout layers position and their respective dropout chance
        @param learning_rate:
        @param weight_decay:  
        @param load_method: Weight loading method. Can be "BucketFS" or "local". Disabled by default
        """
        model = cls(layers=layers,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    l1_strength=l1_strength,
                    l2_strength=l2_strength)

        if weight_path is None:
            weight_path = generate_suffix(layers, 'classifier', load_method)

        if load_method is not None:
            if load_method == "BucketFS":
                # Load weight from BucketFS
                client = BucketFS_client()
                weight = client.download(weight_path)
            elif load_method == "local":
                # Load weight by local file
                with open(weight_path, 'rb') as file:
                    weight = io.BytesIO(file.read())
            model.load_state_dict(torch.load(weight))

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return model, optimizer

    def train(self,model,num_epochs,batch_size,patience,layers,train_loader,val_loader,onehotencoder,scaler, \
            optimizer,scheduler,device,continous_columns,categorical_columns,loss_ratio=(1,1),save=None):
    
        best_loss = float('inf')
        best_state_dict = None
        model.to(device)
        counter = 0
        # Training loop
        for epoch in range(num_epochs):
            train_progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], Training Progress', position=0, leave=True)

            running_loss = 0.0
            running_sample_count = 0.0
            train_predictions = []
            train_targets = []
            for inputs,target,_  in train_progress:
                # Forward pass
                inputs = inputs.to(device)
                outputs = model(inputs)
                # print(Counter(outputs.view(-1).tolist()))
                target = target.to(device)
                # print(Counter(target.view(-1).tolist()))
                # break

                loss = nn.BCEWithLogitsLoss()(outputs,target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics calculation
                predictions = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
                targets = target.cpu().numpy()
                train_predictions.extend(predictions)
                train_targets.extend(targets)

                running_loss += loss.item()*batch_size
                running_sample_count += inputs.shape[0]

            average_loss = running_loss / running_sample_count      # Final loss: multiply by batch size then averaged over all samples
            train_accuracy = accuracy_score(train_targets, train_predictions)
            train_precision = precision_score(train_targets, train_predictions)
            train_recall = recall_score(train_targets, train_predictions)
            train_f1 = f1_score(train_targets, train_predictions)
            
            train_progress.set_postfix({"Training Loss": average_loss})
            train_progress.update()
            train_progress.close()

            # Calculate validation loss
            val_progress = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], Validation Progress', position=0, leave=True)

            val_running_loss = 0.0
            val_running_sample_count = 0.0
            val_predictions = []
            val_targets = []
            for val_inputs, val_target, _  in val_progress:
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs)
                val_target = val_target.to(device)

                val_loss = nn.BCEWithLogitsLoss()(val_outputs,val_target)

                # Metrics calculation
                predictions = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
                targets = target.cpu().numpy()
                val_predictions.extend(predictions)
                val_targets.extend(targets)

                val_running_loss += val_loss.item()*batch_size
                val_running_sample_count += val_inputs.shape[0]

            val_avg_loss = val_running_loss / val_running_sample_count
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision = precision_score(val_targets, val_predictions)
            val_recall = recall_score(val_targets, val_predictions)
            val_f1 = f1_score(val_targets, val_predictions)

            val_progress.set_postfix({"Validation Loss": val_avg_loss})
            val_progress.update()
            val_progress.close()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss  : {average_loss:.8f}, Accuracy: {train_accuracy:.8f}, Precision: {train_precision:.8f}, Recall: {train_recall:.8f}, F1 Score: {train_f1:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_avg_loss:.8f}, Accuracy: {val_accuracy:.8f}, Precision: {val_precision:.8f}, Recall: {val_recall:.8f}, F1 Score: {val_f1:.8f}")

            # Update the learning rate
            scheduler.step()
            print(f"Epoch [{epoch+1}/{num_epochs}]: Learning Rate = {scheduler.get_last_lr()}\n")

            # Check if validation loss has improved
            if val_avg_loss < best_loss - 0.001:
                best_loss = val_avg_loss
                best_state_dict = model.state_dict()
                counter = 0
            else:
                counter += 1
            # Early stopping condition
            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break
            # train_progress.close()

        # Save training weight 
        if (save is not None): 
            model.load_state_dict(best_state_dict)
            layers_str = '_'.join(str(item) for item in layers[1:]) #@TODO: file name hack
            file_name = f'autoencoder_{layers_str}_{loss_ratio}.pth'
            if (save=="BucketFS"):   
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                client = BucketFS_client()
                client.upload(f'autoencoder/{file_name}', buffer)
            elif (save=="local"):
                torch.save(model.state_dict(), file_name)
                print(f'Saved weight to {file_name}')
        else:
            pass
