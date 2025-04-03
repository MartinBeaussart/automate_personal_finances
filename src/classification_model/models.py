from transformers import DistilBertModel
import torch


# TODO try sentence-bert or/and bertopic

class DefaultModel(torch.nn.Module):
    """
    A PyTorch model that uses a pre-trained DistilBERT model for classification tasks.

    Attributes:
        l1 (DistilBertModel): The pre-trained DistilBERT model.
        pre_classifier (torch.nn.Linear): A linear layer to reduce the dimensionality of the output.
        dropout (torch.nn.Dropout): A dropout layer to prevent overfitting.
        classifier (torch.nn.Linear): A final linear layer for classification.

    """
    
    def __init__(self):
        super(DefaultModel, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 300)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(300, 15)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): The input IDs.
            attention_mask (torch.Tensor): The attention mask.

        """
        with torch.no_grad(): 
            output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            
        hidden_state = output_1.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class Bert(torch.nn.Module):
    """
    Pre-trained DistilBERT model.
    """
    
    def __init__(self):
        super(Bert, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): The input IDs.
            attention_mask (torch.Tensor): The attention mask.

        """
        
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0]
            
        return output