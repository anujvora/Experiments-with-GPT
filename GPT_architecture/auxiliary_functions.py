
import torch as tr
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def text_to_token(text,tokenizer):
    tokens = tokenizer.encode(text)
    return tr.tensor((tokens)).unsqueeze(0)

def token_to_text(tokens,tokenizer):
    tokens_list = tokens.squeeze(0).tolist()
    return tokenizer.decode(tokens_list)

# predicted_token_list = input_tokens.squeeze(0).tolist()
# tokenizer.decode(predicted_token_list)
def print_sample_text(input_sentence,tokenizer,gpt_model):
    max_length_text = 20
# input_sentence = "Topic"
# # # input_tokens = tr.tensor()
# new_input = tokenizer.encode(input_sentence)

# # input_tokens.shape
    input_tokens = text_to_token(input_sentence,tokenizer)
    for _ in range(max_length_text):
        output = nn.functional.softmax(gpt_model(input_tokens)[:,-1],dim=-1)
        # print(output.shape)
        new_input = tr.argmax(output)
        # print(input_tokens,new_input.view(1,1))
        input_tokens = tr.cat((input_tokens,new_input.view(1,1)),dim=1)
        # print(new_input,input_tokens)

    print(token_to_text(input_tokens,tokenizer))



def assign(left_val,right_val):
    if left_val.shape != right_val.shape:
        ValueError(f"Left {left_val.shape} and right {right_val.shape} shapes do not match")
    return nn.Parameter(tr.tensor(right_val))

def load_weights_into_model(gpt_model,params):
    gpt_model.Position_Embedding.weight = assign(gpt_model.Position_Embedding.weight,params["wpe"])
    gpt_model.Token_Embedding.weight = assign(gpt_model.Token_Embedding.weight,params["wte"])

    for b in range(len(params["blocks"])):
        ## attention block weights, biases, and projections
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt_model.Transformer_block[b].Multihead_attn.W_query.weight = assign(gpt_model.Transformer_block[b].Multihead_attn.W_query.weight, q_w.T)
        gpt_model.Transformer_block[b].Multihead_attn.W_key.weight = assign(gpt_model.Transformer_block[b].Multihead_attn.W_key.weight, k_w.T)
        gpt_model.Transformer_block[b].Multihead_attn.W_value.weight = assign(gpt_model.Transformer_block[b].Multihead_attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt_model.Transformer_block[b].Multihead_attn.W_query.bias = assign(gpt_model.Transformer_block[b].Multihead_attn.W_query.bias, q_b)
        gpt_model.Transformer_block[b].Multihead_attn.W_key.bias = assign(gpt_model.Transformer_block[b].Multihead_attn.W_key.bias, k_b)
        gpt_model.Transformer_block[b].Multihead_attn.W_value.bias = assign(gpt_model.Transformer_block[b].Multihead_attn.W_value.bias, v_b)

        gpt_model.Transformer_block[b].Multihead_attn.out_proj.weight = assign(gpt_model.Transformer_block[b].Multihead_attn.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt_model.Transformer_block[b].Multihead_attn.out_proj.bias = assign(gpt_model.Transformer_block[b].Multihead_attn.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        ## feedforward layers
        gpt_model.Transformer_block[b].FFN.layers[0].weight = assign(gpt_model.Transformer_block[b].FFN.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt_model.Transformer_block[b].FFN.layers[0].bias = assign(gpt_model.Transformer_block[b].FFN.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])

        gpt_model.Transformer_block[b].FFN.layers[2].weight = assign(gpt_model.Transformer_block[b].FFN.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt_model.Transformer_block[b].FFN.layers[2].bias = assign(gpt_model.Transformer_block[b].FFN.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        ## layer normalizations
        gpt_model.Transformer_block[b].Layer_norm1.scale = assign(gpt_model.Transformer_block[b].Layer_norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt_model.Transformer_block[b].Layer_norm1.shift = assign(gpt_model.Transformer_block[b].Layer_norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt_model.Transformer_block[b].Layer_norm2.scale = assign(gpt_model.Transformer_block[b].Layer_norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt_model.Transformer_block[b].Layer_norm2.shift = assign(gpt_model.Transformer_block[b].Layer_norm2.shift, params["blocks"][b]["ln_2"]["b"])

    ## final layer norm
    gpt_model.Final_Layer_Norm.scale = assign(gpt_model.Final_Layer_Norm.scale, params["g"])
    gpt_model.Final_Layer_Norm.shift = assign(gpt_model.Final_Layer_Norm.shift, params["b"])
    gpt_model.Final_output.weight = assign(gpt_model.Final_output.weight, params["wte"])


def generate(input_sentence, tokenizer, gpt_model, device, context_size, temperature, max_length_text, top_k):
    input_tokens = text_to_token(input_sentence, tokenizer)[:,-context_size:].to(device)
    # print(device)
    for _ in range(max_length_text):
        with tr.no_grad():
            output = gpt_model(input_tokens)[:,-1] #
        if top_k:
            top_k_vals, top_k_ind = tr.topk(output,top_k)
            # print(output.shape,top_k_vals)
            new_output = tr.where(condition=output < top_k_vals[:,-1],
                                  input= tr.tensor(float("-inf")),
                                  other=output)
            # print(new_output[:,top_k_ind])
            temp_scaled_normalized = nn.functional.softmax(new_output/temperature, dim=-1)
            # print(temp_scaled_normalized[:,top_k_ind])
            new_input = tr.multinomial(temp_scaled_normalized, num_samples=1)
        else:
            new_input = tr.argmax(output,keepdim=True)
        input_tokens = tr.cat((input_tokens,new_input),dim=1)

    return token_to_text(input_tokens,tokenizer)


##-----------------------------------------------------------------------------------------

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        super().__init__()  
        self.data = pd.read_csv(csv_file)

        # self.max_length = len(tokenizer.encode(train_df["text"].describe().top))
        # self.data_tensor = tr.ones((self.data.shape[0],self.max_length))*pad_token_id
        self.list_encoded_text = [tokenizer.encode(sms) for sms in self.data["Text"]]
        self.max_length = self.get_max_token_length()

        self.encoded_text = [l + [pad_token_id]*(self.max_length-len(l)) for l in self.list_encoded_text]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        encoded = self.encoded_text[index]
        label = self.data.iloc[index]["Label"]

        return (tr.tensor(encoded, dtype=tr.long),tr.tensor(label,dtype=tr.long))

    def get_max_token_length(self):
        max_len = 0
        for l in self.list_encoded_text:
            if len(l) > max_len:
                max_len = len(l)
        return max_len

##-------------------------------------------------------------
''' Chapter 6 : SMS classification tailerd loss and accuracy functions'''


def calc_accuracy_loader(data_loader, model, num_batches=None):
    model.eval()
    correct_pred, num_examples = 0, 0

    if num_batches == None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))

    with tr.no_grad():
        correct_pred = 0
        for inp, label in data_loader:
            out = model(inp)
            pred_labels_batch = tr.argmax(out[:,-1,:],dim=1)
            correct_pred += (pred_labels_batch == label).sum().item()
            num_examples += len(label)
    
    return correct_pred/num_examples


def calc_loss_batch(input_batch, target_batch, model):
    out = model(input_batch)[:,-1,:]
    loss = nn.functional.cross_entropy(out,target_batch)
    return loss

def calc_loss_loader(data_loader, model, num_batches=None):

    loss = 0

    if num_batches == None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))

    for inp, label in data_loader:
        loss += calc_loss_batch(inp, label, model).item()
    
    return loss/num_batches

def eval_batch_loss(data_loader, model):
    loss = 0
    model.eval()

    # if num_batches == None:
    #     num_batches = len(data_loader)
    # else:
    #     num_batches = min(num_batches,len(data_loader))

    num_batches = len(data_loader)
    for inp, label in data_loader:
        out = model(inp)[:,-1,:]
        loss += nn.functional.cross_entropy(out,label).item()
        
    return loss/num_batches

def eval_loss(train_loader, validation_loader, model):

    train_loss = eval_batch_loss(train_loader, model)
    valid_loss = eval_batch_loss(validation_loader, model)

    return train_loss, valid_loss


def train_model_generic(model, n_epoch, optimizer, train_loader, validation_loader, test_loader):
    loss_epoch = []
    train_acc_list = []
    valid_acc_list = []
    train_losses = []
    valid_losses = []
    for epoch in range(n_epoch):
        model.train()
        loss_batches = 0
        for i, batch_data in enumerate(train_loader):
            inp, label = batch_data
            optimizer.zero_grad()
            loss = calc_loss_batch(inp,label,model) 
            loss.backward()
            optimizer.step()
            loss_batches += loss
            if i%10 == 0:
                print(f"Losses in Epoch: {epoch}, batch no.: {i}")
                train_loss, valid_loss = eval_loss(train_loader, validation_loader, model)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                print(f"Train loss: {train_loss:0.3f}")
                print(f"Validation loss: {valid_loss:0.3f}")

        
        train_acc = calc_accuracy_loader(train_loader,model)
        valid_acc = calc_accuracy_loader(validation_loader,model)
        print(f"accuracies at epoch: {epoch}")
        print(f"Training acc.: {100*train_acc:0.4f} , Validation acc.: {100*valid_acc:0.4f}")
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
        loss_epoch.append(loss_batches)

    return loss_epoch, train_acc, valid_acc, train_losses, valid_losses
