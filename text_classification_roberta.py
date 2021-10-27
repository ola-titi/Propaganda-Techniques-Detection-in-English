#Ola altiti
"""
Original file is located at
    https://colab.research.google.com/drive/1qmwDhnxDZgqsCfIsnBfzeJuJb_3vJPnl
"""

# To determine which version you're using:
!pip show tensorflow

# For the current version: 
!pip install --upgrade tensorflow

pip install transformers

import os
import torch
# Enable cuda to report the error where it occurs.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Commented out IPython magic to ensure Python compatibility.
# Import appropriate libraries

from transformers import BertForSequenceClassification,BartForSequenceClassification, AdamW, BertConfig,Adafactor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification,BertForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer
from transformers import RobertaModel
from torch.optim import Optimizer
from sklearn import preprocessing
from transformers import BertModel
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
# % matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import time
import sys
import csv

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead...')
    device = torch.device("None")

"""##Dataset"""

train_data=pd.read_csv('Train-Propaganda-English.csv')
dev_data=pd.read_csv('Dev-propaganda-English.csv')
test_data=pd.read_csv('Test-propaganda-English.csv')

"""##Processing and Training"""

# Labels encoding 
le1 = preprocessing.LabelEncoder()
le1.fit(train_data.label)

train_data['labels'] = le1.transform(train_data.label)

le2 = preprocessing.LabelEncoder()
le2.fit(dev_data.label)
dev_data['labels']=le2.transform(dev_data.label)

# span: propaganda snippet from a news article
# Label: one propaganda technique out of 14 techniques

train_data=train_data[['span','labels']]
dev_data=dev_data[['span','labels']]

index2label = ['Appeal_to_Authority',
'Appeal_to_fear-prejudice',
'Bandwagon,Reductio_ad_hitlerum',
'Black-and-White_Fallacy',
'Causal_Oversimplification',
'Doubt',
'Exaggeration,Minimisation',
'Flag-Waving',
'Loaded_Language',
'Name_Calling,Labeling',
'Repetition',
'Slogans',
'Thought-terminating_Cliches',
'Whataboutism,Straw_Men,Red_Herring'
]

train_spans = train_data.span.values
train_labels = train_data.labels.values
eval_spans=dev_data.span.values
eval_labels=dev_data.labels.values

# Tokenize all sentences  
def convert_sentence_to_input_feature(sentence, tokenizer, add_cls_sep=True, max_seq_len=120):
  tokenized_sentence = tokenizer.encode_plus(sentence,
                                             add_special_tokens=add_cls_sep,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,truncation=True)
  
  
  return tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']

def get_data(spans, techniques=None):
  
  s_attention_masks = []
  s_inputs = []
  for i, sentence in enumerate(spans):
    s_input_ids, s_mask = convert_sentence_to_input_feature(sentence,tokenizer)
    s_inputs.append(s_input_ids)
    s_attention_masks.append(s_mask)
  

  max_sent_len = 0
  for sent in spans:
    sent_len = len(sent.split(' '))
    max_sent_len = max(max_sent_len, sent_len)
 
  print(max_sent_len)

  s_inputs = torch.tensor(s_inputs)
  labels = torch.tensor(techniques)
  s_masks = torch.tensor(s_attention_masks)
  tensor_data = TensorDataset(s_inputs, labels, s_masks)
  # Use DataLoader to save on memory during training 
  dataloader = DataLoader(tensor_data, sampler = RandomSampler(tensor_data),batch_size=4)
  return dataloader

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of the predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Initialize RoBERTa tokenizer from transformer library to tokenize the sentences

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', lower_case=True)

# Get PyTorch tensors
train_dataloader = get_data(train_spans, train_labels)
eval_dataloader = get_data(eval_spans, eval_labels)

# Set the seed value
seed_val = 1000
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
epochs = 3

# Load RobertaForSequenceClassification, the pretrained RoBERTa model with a single linear classification layer on top.
modelR=RobertaForSequenceClassification.from_pretrained("roberta-large",num_labels=14) # Use the 24-layer RoBERTa model


optimizer = Adafactor(modelR.parameters(),lr = 1e-5,
                      eps = (1e-3,1e-3),
                      weight_decay=0.1,
                      decay_rate =0.8,
                      clip_threshold=1.0,
                      relative_step=False,
                       scale_parameter=False)

# to run the model on the GPU. 
modelR.cuda()

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# train the model in the GPU device
modelR.to(device)
training_stats=[]
loss_values = []
total_t0 = time.time()
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    modelR.train()
    for step, batch in enumerate(train_dataloader):
      if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
      b_input_ids = batch[0].to(device)
      b_labels = batch[1].to(device)
      b_input_mask = batch[2].to(device)
      torch.cuda.empty_cache()
      modelR.zero_grad()        
      loss, logits = modelR(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      print(loss,"\t",logits)
      total_train_loss += loss.item()
      
      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(modelR.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      # The optimizer dictates the "update rule"--how the parameters are
      # modified based on their gradients, the learning rate, etc.
      optimizer.step()

        # Update the learning rate.
      scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    modelR.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in eval_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[1].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = modelR(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(eval_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

#print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()

# Create sentence and label lists
sentences = dev_data.span.values


# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 120,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
#labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler)

len(prediction_dataloader)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
modelR.eval()

# Tracking variables 
predictions = []
total_predictions=[]
# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  #print(batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = modelR(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  #true_labels.append(label_ids)

print('DONE.')

"""#Evaluation"""

pred_label=[]
for i in range(len(predictions)):
  pred_label.append(np.argmax(predictions[i]))

results = []
for i in range(len(pred_label)):
  results.append(index2label[pred_label[i]])

act_y=dev_data['label']
print("Micro f1 :",f1_score(act_y, result, average='micro'))
print("Accuracy is:",accuracy_score(act_y,result))
print(metrics.classification_report(act_y, result))
print("Accuracy is:",accuracy_score(act_y,result))
cm(act_y,result)
plot(history)

#Write predictions to txt file  
test_ids=test_data['id']
test_s=test_data['start']
test_end=test_data['end']
with open("predictions-ensemble.txt", "w") as fout:
        for article_id, prediction, span_start, span_end in zip(test_ids, results, test_s,test_end):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))