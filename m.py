# pip3 install torchviz
# sudo apt install graphviz
# pip3 install graphviz
# pip3 install matplotlib
# pip3 install ptflops

from torch import nn, optim
from torchviz import make_dot
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time, copy, torch
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from ptflops import get_model_complexity_info

#
os.chdir('/home/ubuntu/lab2')

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
def read_bci_data():
    S4b_train = np.load('data/S4b_train.npz')
    X11b_train = np.load('data/X11b_train.npz')
    S4b_test = np.load('data/S4b_test.npz')
    X11b_test = np.load('data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)
    train_data, train_label, test_data, test_label = torch.from_numpy(train_data).double(), torch.from_numpy(
        train_label).double(), torch.from_numpy(test_data).double(), torch.from_numpy(test_label).double()
    # print('Data shapes:', train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data, train_label, test_data, test_label


# Get dataloader
def get_dataloader():
    tr_x, tr_y, te_x, te_y = read_bci_data()
    tr_dataset = TensorDataset(tr_x, tr_y)
    te_dataset = TensorDataset(te_x, te_y)
    return DataLoader(dataset=tr_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(dataset=te_dataset,
                                                                                           batch_size=BATCH_SIZE,
                                                                                           shuffle=False)


# Plot model
def plot_model(model_, inputs_, device_, output_path: str = 'nn'):
    y_ = model_.to(device_)(inputs_.to(device_))
    # dot = make_dot(y_.mean(), params=dict(model_.named_parameters()), show_attrs=True, show_saved=True)
    dot = make_dot(y_.mean(), params=dict(model_.named_parameters()))
    dot.render(output_path, format='png', view=False)


# EEGNet
class EEGNet(nn.Module):
    def __init__(self, activation: str):
        if activation == 'relu':
            self.activation_class = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation_class = nn.LeakyReLU
        elif activation == 'elu':
            self.activation_class = nn.ELU
        else:
            raise ValueError('Parameter of activation must be relu, leaky_relu, or elu.')
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16)
        )
        self.depthwise_conv_front = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32),
        )
        self.depthwise_conv_activation = self.activation_class()
        self.depthwise_conv_end = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.separable_conv_front = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32),
        )
        self.separable_conv_activation = self.activation_class()
        self.separable_conv_end = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthwise_conv_front(x)
        x = self.depthwise_conv_activation(x)
        x = self.depthwise_conv_end(x)
        x = self.separable_conv_front(x)
        x = self.separable_conv_activation(x)
        x = self.separable_conv_end(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)
        return x


#
class DeepConvNet(nn.Module):
    def __init__(self, activation: str):
        if activation == 'relu':
            self.activation_class = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation_class = nn.LeakyReLU
        elif activation == 'elu':
            self.activation_class = nn.ELU
        else:
            raise ValueError('Parameter of activation must be relu, leaky_relu, or elu.')
        super().__init__()

        self.c = 2
        self.t = 750
        self.n = 2
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (self.c, 1)),
            nn.BatchNorm2d(25),
        )
        self.act1 = self.activation_class()
        self.seq2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50)
        )
        self.act2 = self.activation_class()
        self.seq3 = nn.Sequential(
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100)
        )
        self.act3 = self.activation_class()
        self.seq4 = nn.Sequential(
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200)
        )
        self.act4 = self.activation_class()
        self.seq5 = nn.Sequential(
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),
        )
        self.dense = nn.Linear(200 * 1 * 43, self.n)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seq1(x)
        x = self.act1(x)
        x = self.seq2(x)
        x = self.act2(x)
        x = self.seq3(x)
        x = self.act3(x)
        x = self.seq4(x)
        x = self.act4(x)
        x = self.seq5(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        # x = self.softmax(x)
        return x


# Training
def train_model(dataloader_train_valid, model_save_dir, model_, criterion_, optimizer_, num_epochs,
                device_, save_epoch_record: bool = False, name_suffix: str = '', sched=None):
    model_ = model_.to(device_)
    start = time.time()
    train_results = []
    valid_results = []
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0
    all_pred = []
    all_labels = []

    for epoch in range(num_epochs):
        print(
            'Epoch {}/{} (lr={})'.format(epoch + 1, num_epochs, sched.optimizer.param_groups[0]['lr'] if sched else ''))
        print('-' * 10)

        start_elapsed = datetime.now()

        all_pred = []
        all_labels = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model_.train()  # Set model_ to training mode
            else:
                model_.eval()  # Set model_ to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            datasetsize = 0

            # Iterate over data.
            for inputs, labels in dataloader_train_valid[phase]:

                datasetsize += labels.shape[0]

                inputs = inputs.to(device_)
                labels = labels.long()

                # print(labels)
                labels = labels.to(device_)

                # zero the parameter gradients
                optimizer_.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # print("########### output ##########")
                    outputs = model_(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion_(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'valid':
                    all_pred.append(preds.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())

            epoch_loss = running_loss / datasetsize
            epoch_acc = running_corrects.double() / datasetsize

            if phase == 'train':
                train_results.append([epoch_loss, epoch_acc])
                if sched:
                    sched.step(epoch_loss)
            if phase == 'valid':
                valid_results.append([epoch_loss, epoch_acc])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model_ (Early Stopping) and Saving our model_, when we get best accuracy
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_.state_dict())
                model_save_name = type(model_).__name__ + "_BestModel_" + name_suffix + ".pt"
                path = os.path.join(model_save_dir, model_save_name)
                torch.save(model_.state_dict(), path)
                #
                # print(type(all_pred))
                # print(all_pred)
                # print(type(all_labels))
                # print(all_labels)
                # plt.figure(figsize=(6, 4))
                # cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_pred), labels=np.arange(1, 11))
                # sn.heatmap(cm, annot=True, cmap=plt.cm.Blues,
                #            xticklabels=np.arange(1, 11),
                #            yticklabels=np.arange(1, 11), )
                # plt.show()

        end_elapsed = datetime.now()

        elapsed_time = end_elapsed - start_elapsed
        print("elapsed time: {}".format(elapsed_time))

        print()

    # Calculating time it took for model_ to train
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model_ weights
    model_.load_state_dict(best_model_wts)

    # Save records of each epoch
    if save_epoch_record:
        with open(os.path.join('epoch_record/', type(model_).__name__) + '_' + name_suffix + '.txt', 'w') as file:
            # First line: train loss
            # Second line: train acc
            # Third line: valid loss
            # Fourth line: valid acc
            train_loss = [item[0] for item in train_results]
            train_acc = [item[1].tolist() for item in train_results]
            valid_loss = [item[0] for item in valid_results]
            valid_acc = [item[1].tolist() for item in valid_results]
            file.write(' '.join(map(str, train_loss)) + '\n')
            file.write(' '.join(map(str, train_acc)) + '\n')
            file.write(' '.join(map(str, valid_loss)) + '\n')
            file.write(' '.join(map(str, valid_acc)) + '\n')

    return model_, train_results, valid_results, all_pred, all_labels


# Settings
MODEL_SAVE_DIR = 'saved_models/'

# Hyper-parameter
# Optimizer: Adam, ...
# Loss function: torch.nn.CrossEntropyLoss(), ...
LR = 0.1
BATCH_SIZE = 400
EPOCHS = 200

#
tr_dataloader, te_dataloader = get_dataloader()

# Single train #######################################################
for act__ in ['relu', 'leaky_relu', 'elu']:
    # act__ = 'elu'
    model = EEGNet(act__).double()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.6,
                                                     min_lr=0.0000001, threshold=1e-4, cooldown=10, verbose=True)
    model, tr_results, va_results, last_epoch_pred, last_epoch_labels = train_model(
        dataloader_train_valid={'train': tr_dataloader, 'valid': te_dataloader},
        model_save_dir=MODEL_SAVE_DIR,
        model_=model,
        criterion_=criterion,
        optimizer_=optimizer,
        num_epochs=EPOCHS,
        save_epoch_record=False,
        name_suffix=act__,
        device_=device,
       sched=scheduler
    )

# Train EEGNet ###############################################################################
for act in ['relu', 'leaky_relu', 'elu']:
    # for act in ['relu']:
    model = EEGNet(act).double()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model, tr_results, va_results, last_epoch_pred, last_epoch_labels = train_model(
        dataloader_train_valid={'train': tr_dataloader, 'valid': te_dataloader},
        model_save_dir=MODEL_SAVE_DIR,
        model_=model,
        criterion_=criterion,
        optimizer_=optimizer,
        num_epochs=EPOCHS,
        save_epoch_record=False,
        name_suffix=act,
        device_=device
    )

# Train DeepConvNet
for act in ['relu', 'leaky_relu', 'elu']:
    model = DeepConvNet(act).double()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model, tr_results, va_results, last_epoch_pred, last_epoch_labels = train_model(
        dataloader_train_valid={'train': tr_dataloader, 'valid': te_dataloader},
        model_save_dir=MODEL_SAVE_DIR,
        model_=model,
        criterion_=criterion,
        optimizer_=optimizer,
        num_epochs=EPOCHS,
        save_epoch_record=True,
        name_suffix=act,
        device_=device
    )

#
# model = EEGNet('elu').double()
# model = DeepConvNet('elu').double()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
#
# model, train_results, valid_results, last_epoch_pred, last_epoch_labels = train_model(
#     dataloader_train_valid={'train': tr_dataloader, 'valid': te_dataloader},
#     model_save_dir=MODEL_SAVE_DIR,
#     model_=model,
#     criterion_=criterion,
#     optimizer_=optimizer,
#     num_epochs=EPOCHS,
#     save_epoch_record=True,
#     name_suffix='elu',
#     device_=device
# )

model___ = EEGNet('relu').double()
plot_model(model___, read_bci_data()[0][0].unsqueeze(dim=1), device, output_path='output/EEGNet_relu')
model___ = DeepConvNet('relu').double()
plot_model(model___, read_bci_data()[0][0].unsqueeze(dim=1), device, output_path='output/DeepConvNet_relu')

# Plot acc-epoch
colors = ['C0', 'C1', 'C2']
# for model_name in ['EEGNet', 'DeepConvNet']:
for model_name in ['EEGNet']:
    max_epoch = 0
    plt.figure(figsize=(8, 6))
    plt.title(f'Activation Function Comparison ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    # plt.yticks(np.arange(10)/10)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    filenames = []
    for root, dirs, files in os.walk('epoch_record'):
        for file in files:
            if model_name in file:
                filenames.append(os.path.join(root, file))
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as file:
            lines = file.read().split('\n')
            train_acc = list(map(float, lines[1].split(' ')))
            test_acc = list(map(float, lines[3].split(' ')))
            max_epoch = max(len(train_acc), max_epoch)
            plt.plot(np.arange(len(train_acc)) + 1, train_acc, label=filename.split('_')[-1].split('.')[0] + '_train',
                     color=colors[i], linestyle='solid')
            plt.plot(np.arange(len(test_acc)) + 1, test_acc, label=filename.split('_')[-1].split('.')[0] + '_test',
                     color=colors[i], linestyle='dashed')
    plt.xticks(np.arange(max_epoch, step=max_epoch // 10) + 2)
    plt.legend()
    plt.savefig('output/' + model_name + '_AccEpoch.png', format='png')
    plt.show()

print('done')

# Test model
TEST_MODE = 'single'  # 'full' or 'single'
MODEL_DIR = 'BestModels'
for model_name in ['EEGNet', 'DeepConvNet']:
    for act in ['relu', 'leaky_relu', 'elu']:
        loaded_model = EEGNet(act).double() if model_name == 'EEGNet' else DeepConvNet(act).double()
        filename = os.path.join(MODEL_DIR, model_name + '_BestModel_' + act + '.pt')
        loaded_model.load_state_dict(torch.load(filename))
        loaded_model.eval()
        #
        bci_data_ = read_bci_data()
        te_x_ = bci_data_[2].to(device)
        te_y_ = bci_data_[3].to(device)
        #
        loaded_model.to(device)
        if TEST_MODE == 'single':
            n_correct = 0
            for sample_, label_ in zip(te_x_, te_y_):
                pred = torch.max(loaded_model(sample_.unsqueeze(dim=1)), 1)[1][0]
                n_correct += 1 if pred == label_ else 0
            test_acc = n_correct / te_y_.size()[0]
        # Full-batch inference
        elif TEST_MODE == 'full':
            _, pred = torch.max(loaded_model(te_x_), 1)
            test_acc = torch.eq(te_y_, pred).sum() / te_y_.size()[0]
        else:
            raise ValueError(f'Invalid TEST_MODE "{TEST_MODE}".')
        print(f'{model_name} + {act}: {test_acc:.4f}')
        # Release memory
        del loaded_model, te_x_, te_y_, pred, test_acc
        torch.cuda.empty_cache()


def test_a_model(filename_: str, model_name_, act_: str):  # single sample inference mode
    loaded_model_ = EEGNet(act_).double() if model_name_ == 'EEGNet' else DeepConvNet(act_).double()
    loaded_model_.load_state_dict(torch.load(filename_))
    loaded_model_.eval()
    #
    _, _, te_x__, te_y__ = read_bci_data()
    te_x__, te_y__ = te_x__.to(device), te_y__.to(device)
    loaded_model_.to(device)
    n_correct_ = 0
    for sample__, label__ in zip(te_x__, te_y__):
        pred_ = torch.max(loaded_model_(sample__.unsqueeze(dim=1)), 1)[1][0]
        n_correct_ += 1 if pred_ == label__ else 0
    test_acc_ = n_correct_ / te_y__.size()[0]
    print(f'{model_name_} + {act_}: {test_acc_:.3f}')
    # Release memory
    del loaded_model_, te_x__, te_y__, pred_, test_acc_
    torch.cuda.empty_cache()


# Test a specific model
test_a_model('BestModels/EEGNet_BestModel_relu.pt', 'EEGNet', 'relu')
test_a_model('BestModels/EEGNet_BestModel_leaky_relu.pt', 'EEGNet', 'leaky_relu')

# Complexity
with torch.cuda.device(0):
    net = EEGNet('relu')
    macs, params = get_model_complexity_info(net, (1, 2, 750), as_strings=True,
                                             print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:}'.format('Computational complexity: ', macs))
    print('{:<30}  {:}'.format('Number of parameters: ', params))
    net = DeepConvNet('relu')
    macs, params = get_model_complexity_info(net, (1, 2, 750), as_strings=True,
                                             print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:}'.format('Computational complexity: ', macs))
    print('{:<30}  {:}'.format('Number of parameters: ', params))
