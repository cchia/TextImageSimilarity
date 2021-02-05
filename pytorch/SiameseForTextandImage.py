import pandas as pd, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import time, datetime, shutil, os
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler, random_split
from transformers import BertModel, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from torch.autograd import Variable
import matplotlib.pyplot as plt

BATCHSIZE = 8
DATAPATH = 'path_to_data_source'
torch.manual_seed(1)

# If there's a GPU available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are {} GPU(s) available'.format(torch.cuda.device_count()))
    print('We will use the GPU: {}'.torch.cuda.get_device_name(0))
# If not
else:
    print('No GPU available, using CPU instead')
    device = torch.device('cpu')

training_stats = []
# Measure the total training time for the whole run
total_t0 = time.time()

df_clean = pd.read_csv(DATAPATH)
df_train = df_clean[df_clean['modelgroup']=='TRAIN']
df_test = df_clean[df_clean['modelgroup']=='TEST']

imagePreprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

IMAGE1FOLDER = '/data1/image1/'
IMAGE2FOLDER = '/data1/image2/'

## Prepare dataset for dataloader
class DatasetForSiamese(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        image1Path = IMAGE1FOLDER+str(self.df.iloc[index, 3])
        image2Path = IMAGE2FOLDER+str(self.df.iloc[index, 4])
        title1 = self.df.iloc[index, 58]
        title2 = self.df.iloc[index, 59]
        price1 = self.df.iloc[index, 62]
        price2 = self.df.iloc[index, 63]
        label = self.df.iloc[index, 6]

        image1 = Image.open(image1Path)
        image2 = Image.open(image2Path)
        # Transform to Tensor
        title1_encoded_dict = tokenizer.encode_plus(
            title1, #Sequence to encode
            max_length = 180, # Pad and truncate all sentences
            pad_to_max_length = True,
            return_attention_mask = True, # construct attention masks
            return_tensors = 'pt' # return pytorch tensors
        )

        title2_encoded_dict = tokenizer.encode_plus(
            title2, #Sequence to encode
            max_length = 180, # Pad and truncate all sentences
            pad_to_max_length = True,
            return_attention_mask = True, # construct attention masks
            return_tensors = 'pt' # return pytorch tensors
        )

        title1_tensor = title1_encoded_dict['input_ids'].squeeze(0)
        title2_tensor = title2_encoded_dict['input_ids'].squeeze(0)
        title1_attn_mask = title1_encoded_dict['attention_mask'].squeeze(0)
        title2_attn_mask = title2_encoded_dict['attention_mask'].squeeze(0)
        image1_tensor = imagePreprocess(image1)
        image2_tensor = imagePreprocess(image2)

        sample = {}
        sample['title1'] = title1
        sample['title2'] = title2
        sample['title1_tensor'] = title1_tensor
        sample['title2_tensor'] = title2_tensor
        sample['title1_attn_mask'] = title1_attn_mask
        sample['title2_attn_mask'] = title2_attn_mask
        sample['price1'] = price1
        sample['price2'] = price2
        sample['image1_tensor'] = image1_tensor
        sample['image2_tensor'] = image2_tensor
        sample['label'] = label
        return sample

    def __len__(self):
        return len(self.df)

train_dataset = DatasetForSiamese(df=df_train)
test_dataset = DatasetForSiamese(df=df_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Divide the dataset by randomly selecting samples
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = BATCHSIZE
# indices = torch.randperm(len(train_dataset)).tolist()
# train_idx, valid_idx = indices[:train_size], indices[train_size:]

# We'll take training samples in random order
train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size = batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset),
    batch_size = batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler = SequentialSampler(test_dataset),
    batch_size = batch_size
)


### Embedder for Text and Image
imageModel = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
textModel = BertModel.from_pretrained('bert-base-multilingual-cased')

def imageEmbedder(imageTensor):
    with torch.no_grad():
        imageModel.cuda()
        output = imageModel(imageTensor).to('cuda')
    return output

def textEmbedder(textTensor, attention_mask):
    if torch.cuda.is_available():
        textTensor = textTensor.to('cuda')
        attention_mask = attention_mask.to('cuda')
        textModel.to('cuda')

    with torch.no_grad():
        output = textModel(input_ids = textTensor, attention_mask = attention_mask)
    return output[0] # the last hidden state is the first element of the output tuple

## Define Constrative Loss for Siamese
class ConstrativeLoss(torch.nn.Module):
    """
    Constrative loss function
    Based on http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ConstrativeLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        avg_distance = euclidean_distance
        loss_constrative = torch.mean(torch.diagonal((1-label) * torch.pow(avg_distance, 2), 0) +
                                      torch.diagonal((label) * torch.pow(torch.clamp(self.margin - avg_distance, min=0.0), 2), 0))
        return loss_constrative

## Network Structure
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.fc1 = nn.Linear(138240, 1000)
        self.fc2 = nn.Linear(2000, 8)

    def forward_once(self, text, image, attn_mask):
        with torch.no_grad():
            imageVector = imageEmbedder(image)
            textVector = textEmbedder(text, attn_mask)
        textVector = self.fc1(torch.flattern(textVector, start_dim=1))
        combineVector = torch.cat((textVector, imageVector), 1)
        output = self.fc2(combineVector)

        return output

    def forward(self, text1, image1, text2, image2, attn_mask1, attn_mask2):
        #Forward pass
        output1 = self.forward_once(text1, image1, attn_mask1)
        output2 = self.forward_once(text2, image2, attn_mask2)
        return (output1, output2)

    def getTextVector(self, text, attn_mask):
        with torch.no_grad():
            textVector1 = textEmbedder(text, attn_mask)
        return textVector1

    def getImageVector(self, image):
        with torch.no_grad():
            imageVector1 = imageEmbedder(image)
        return imageVector1

siameseModel = Siamese()
siameseModel.cuda()

## Optimizer definition
EPOCH = 3
epochs = EPOCH

optimizer = torch.optim.Adam(siameseModel.parameters(), lr=0.0005)

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # default value in run_glue.py
                                            num_training_steps = total_steps)
# loss_fn = nn.BCELoss()
loss_fn = ConstrativeLoss()

def flat_accuracy(preds, labels):
    return np.sum(preds == labels)/len(labels)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    bestpath = '/data/bestPath/'
    torch.save(state, os.path.join(directory, filename))
    if is_best:
        if not os.path.exists(bestpath):
            os.makedirs(bestpath)
        shutil.copyfile(os.path.join(directory, filename), os.path.join(bestpath, 'model_best.pth.tar'))

iteration_number = 0
counter = []
loss_history = []
best_acc = 0.0
total_t0 = time.time()

for epoch in range(epochs):
    t0 = time.time()

    # Reset the total loss for this epoch
    total_train_loss = 0
    siameseModel.train()

    for i, data in enumerate(train_dataloader, 0):
        title1_tensor = data['title1'].cuda()
        title2_tensor = data['title2'].cuda()
        title1_attn_mask = data['title1_attn_mask'].cuda()
        title2_attn_mask = data['title2_attn_mask'].cuda()
        price1 = data['price1'].cuda()
        price2 = data['price2'].cuda()
        image1_tensor = data['image1_tensor'].cuda()
        image2_tensor = data['image2_tensor'].cuda()
        label = data['label'].cuda()
        optimizer.zero_grad()
        siameseModel.zero_grad()

        output1, output2 = siameseModel(title1_tensor, image1_tensor, title2_tensor, image2_tensor, title1_attn_mask, title2_attn_mask)
        loss_constrative = loss_fn(output1, output2)

        if i%10==0: print("finishing batch {}".format(i))
        print("training loss = {0: .2f".format(loss_constrative.item()))
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. Loss is a tensor containing a
        # single value; the .item() function just returns the Python value
        # from the tensor.
        total_train_loss += loss_constrative.item()

        loss_constrative.backward()

        # Clip the norm of the gradients to 1.0
        # This is to help prevent the 'exploding gradients' problem
        torch.nn.utils.clip_grad_norm_(siameseModel.parameters(), 1.0)

        # Update the parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule" -- how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took
    training_time = format_time(time.time() - t0)

    print("")
    print("Average training loss: {0: 0.2f}".format(avg_train_loss))
    print("Training epoch took: {:}".format(training_time))

    # Validation
    # After the completion of each training epoch, measure performance on validation set

    print("")
    print("Running validation...")

    t0 = time.time()
    siameseModel.eval()

    # Tracking varaibles
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for iv, datav in enumerate(validation_dataloader, 0):
        title1_tensor_v = datav['title1'].cuda()
        title2_tensor_v = datav['title2'].cuda()
        title1_attn_mask_v = datav['title1_attn_mask'].cuda()
        title2_attn_mask_v = datav['title2_attn_mask'].cuda()
        price1_v = datav['price1'].cuda()
        price2_v = datav['price2'].cuda()
        image1_tensor_v = datav['image1_tensor'].cuda()
        image2_tensor_v = datav['image2_tensor'].cuda()
        label_v = datav['label'].cuda()

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the logits output by the model. The logits are the output values
            # prior to applying an activation function like the softmax
            output1, output2 = siameseModel(title2_tensor_v, image1_tensor_v, title2_tensor_v, image2_tensor_v, title1_attn_mask_v, title2_attn_mask_v)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        euclidean_distance = euclidean_distance.to('cpu').numpy()
        label_v = label_v.to('cpu').numpy()
        prediction = [1 if i<=1 else 0 for i in euclidean_distance]
        total_eval_accuracy += flat_accuracy(prediction, label_v)

        loss_constrative = loss_fn(output1, output2, label_v)
        total_eval_loss += loss_constrative

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(" Accuracy: {0:.2f".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    #Measure how long the validation run took
    validation_time = format_time(time.time() - t0)

    print("Validation loss: {0:.2f}".format(avg_val_loss))
    print("Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch
    training_stats.append(
        {
            'epoch': epoch+1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    is_best = avg_val_accuracy > best_acc
    best_acc = max(avg_val_accuracy, best_acc)

    output_dir = 'data/output/epoch{}/'.format(epoch+1)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to {}".format(output_dir))

    # Save a trained model, configuration and tokenizer using `save_pretrained()`
    # They can then be reloaded using 'from_pretrained()'

    save_checkpoint({
        'epoch': epoch+1,
        'state_dict': siameseModel.state_dict(),
        'best_acc1': best_acc,
        'optimzizer': optimizer.state_dict()
    }, is_best, output_dir)

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time-time()-total_t0)))
