import os
import sys
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))
from mcm.mcm import download_latest_model
from mcm.mcm import get_mode_torch
from nnmodels import NPOptionsNet
from ImgGenerator import ImgGenerator

mode_torch = get_mode_torch()

def imshow(img):
    """
    # functions to show an image
    """
    import matplotlib.pyplot as plt
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class OptionsDetector(ImgGenerator):
    """
    TODO: describe class
    """
    def __init__(self, options = {}):
        """
        TODO: describe __init__
        """
        # input
        self.HEIGHT         = 64
        self.WEIGHT         = 295
        self.COLOR_CHANNELS = 3

        # outputs 1
        self.CLASS_REGION = options.get("class_region", [
            "xx-unknown",
            "eu-ua-2015",
            "eu-ua-2004",
            "eu-ua-1995",
            "eu",
            "xx-transit",
            "ru",
            "kz",
            "eu-ua-ordlo-dpr",
            "eu-ua-ordlo-lpr",
            "ge",
            "by",
            "su",
            "kg"
        ])

        # outputs 2
        self.CLASS_STATE = options.get("class_state", [
            "garbage",
            "filled",
            "not filled",
            "empty"
        ])

        # model
        self.MODEL = None

        # train hyperparameters
        self.BATCH_SIZE       = 32
        self.EPOCHS           = 150

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def create_model(self):
        """
        TODO: describe method
        """
        self.MODEL = NPOptionsNet()
        if mode_torch == "gpu":
            self.MODEL = self.MODEL.cuda()
        return self.MODEL

    def prepare(self, base_dir, verbose=1):
        """
        TODO: describe method
        """
        if verbose:
            print("START PREPARING")
        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        # compile generators
        self.train_generator = self.compile_train_generator(train_dir,
            (
                self.HEIGHT,
                self.WEIGHT
            ),
            self.BATCH_SIZE)
        self.validation_generator = self.compile_test_generator(validation_dir,
            (
                self.HEIGHT,
                self.WEIGHT
            ),
            self.BATCH_SIZE)
        self.test_generator = self.compile_test_generator(test_dir,
            (
                self.HEIGHT,
                self.WEIGHT
            ),
            self.BATCH_SIZE)

        if verbose:
            print("DATA PREPARED")

    def train(self, log_dir="./log", load_model=None, with_aug=0, verbose=1):
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        from tqdm import tqdm

        if not os.path.exists(log_dir):
            os.mkdirs(log_dir)
        # init count outputs
        self.create_model()
        if load_model is not None:
            self.load(load_model)
        criterion_reg = nn.CrossEntropyLoss()
        criterion_line = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(self.MODEL.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-07)
        history = []

        loss = 0.0
        loss_reg = 0.0
        loss_line = 0.0
        acc = 0
        acc_reg = 0
        acc_line = 0
        
        g_loss = 0.0
        g_loss_reg = 0.0
        g_loss_line = 0.0
        g_acc = 0
        g_acc_reg = 0
        g_acc_line = 0
        best_val_acc = 0.0
        
        for epoch in range(self.EPOCHS):  # loop over the dataset multiple times
            self.train_generator.rezero()
            trainGenerator = self.train_generator.generator(with_aug=with_aug)
            self.validation_generator.rezero()
            validationGenerator = self.validation_generator.generator()
            diplay_per = int(self.train_generator.batch_count/50) or 1
            train_bar = tqdm(enumerate(trainGenerator, 0), total=self.train_generator.batch_count)
            for i, data in train_bar:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                #print("inputs", inputs)
                #break
                inputs = torch.from_numpy(inputs)
                if mode_torch == "gpu":
                    inputs = inputs.cuda()
                outputs = self.MODEL(inputs)
                #print("outputs", outputs)
                #print("labels", labels)
                label_reg = torch.from_numpy(labels[0])
                label_cnt = torch.from_numpy(labels[1])
                if mode_torch == "gpu":
                    label_reg = label_reg.cuda()
                    label_cnt = label_cnt.cuda()
                loss_reg = criterion_reg(outputs[0], torch.max(label_reg, 1)[1])
                loss_line = criterion_line(outputs[1], torch.max(label_cnt, 1)[1])
                loss = (loss_reg + loss_line)/2
                
                acc_reg = (torch.max(outputs[0], 1)[1] == torch.max(label_reg, 1)[1]).float().sum()/self.BATCH_SIZE
                acc_line = (torch.max(outputs[1], 1)[1] == torch.max(label_cnt, 1)[1]).float().sum()/self.BATCH_SIZE
                acc = (acc_reg+acc_line)/2

                loss.backward()
                optimizer.step()
                
                g_loss += loss
                g_loss_reg += loss_reg
                g_loss_line += loss_line
                g_acc += acc
                g_acc_reg += acc_reg
                g_acc_line += acc_line
                # print statistics
                if i%diplay_per == 0:
                    g_loss /= diplay_per
                    g_loss_reg /= diplay_per
                    g_loss_line /= diplay_per
                    g_acc /= diplay_per
                    g_acc_reg /= diplay_per
                    g_acc_line /= diplay_per
                    
                    train_bar.set_description(f'[TRAIN {epoch + 1}, {i + 1}] loss: {g_loss} loss_reg: {g_loss_reg} loss_line: {g_loss_line} acc: {g_acc} acc_reg: {g_acc_reg} acc_line: {g_acc_line}')
                    g_loss = 0.0
                    g_loss_reg = 0.0
                    g_loss_line = 0.0
                    g_acc = 0
                    g_acc_reg = 0
                    g_acc_line = 0
            # validation
            val_acc, val_acc_reg, val_acc_line \
                            = self.test(testGenerator=validationGenerator, verbose=0)
            print(f'[VALIDATION {epoch + 1}]',
                    f'val_acc: {val_acc} '
                    f'val_acc_reg: {val_acc_reg} '
                    f'val_acc_line: {val_acc_line} '
                    )
            # save after each epochs
            model_path = os.path.join(
                log_dir,
                f"best.pb"
            )
            if val_acc > best_val_acc:
                torch.save(self.MODEL.state_dict(), model_path)
        print('Finished Training')
        return  self.MODEL

    def test(self, testGenerator=None, verbose=1):
        """
        TODO: describe method
        """
        if testGenerator == None:
            testGenerator = self.test_generator.generator()
        acc = 0
        acc_reg = 0
        acc_line = 0
        all_acc = 0
        all_acc_reg = 0
        all_acc_line = 0
        n = 0
        for i, data in enumerate(testGenerator, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # test
            #print(i, inputs.shape)
            inputs = torch.from_numpy(inputs)
            if mode_torch == "gpu":
                inputs = inputs.cuda()
            outputs = self.MODEL(inputs)
            label_reg = torch.from_numpy(labels[0])
            label_cnt = torch.from_numpy(labels[1])
            if mode_torch == "gpu":
                label_reg = label_reg.cuda()
                label_cnt = label_cnt.cuda()
            
            acc_reg = (torch.max(outputs[0], 1)[1] == torch.max(label_reg, 1)[1]).float().sum()
            acc_line = (torch.max(outputs[1], 1)[1] == torch.max(label_cnt, 1)[1]).float().sum()
            acc = (acc_reg+acc_line)/2
            
            all_acc += acc.cpu().numpy()
            all_acc_reg += acc_reg.cpu().numpy()
            all_acc_line += acc_line.cpu().numpy()
            
            n += 1

        all_acc /= n*self.BATCH_SIZE
        all_acc_reg /= n*self.BATCH_SIZE
        all_acc_line /= n*self.BATCH_SIZE
        return all_acc, all_acc_reg, all_acc_line

    def save(self, path, verbose=1):
        """
        TODO: describe method
        """
        if self.MODEL != None:
            if bool(verbose):
                print("model save to {}".format(path))
            torch.save(self.MODEL.state_dict(), path)

    def isLoaded(self):
        """
        TODO: describe method
        """
        if self.MODEL == None:
            return False
        return True

    def load(self, path_to_model="latest", options={}, verbose = 0):
        """
        TODO: describe method
        """
        self.create_model()
        if path_to_model == "latest":
            model_info   = download_latest_model(self.get_classname(), "simple")
            path_to_model   = model_info["path"]
            options["class_region"] = model_info["class_region"]

        self.CLASS_REGION = options.get("class_region", ["xx-unknown", "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu", "xx-transit", "ru", "kz", "eu-ua-ordlo-dpr", "eu-ua-ordlo-lpr", "ge", "by", "su", "kg"])

        if mode_torch == "gpu":
            self.MODEL.load_state_dict(torch.load(path_to_model))
        else:
            self.MODEL.load_state_dict(torch.load(path_to_model,  map_location=torch.device('cpu')))
        self.MODEL.eval()
    
    @torch.no_grad()
    def predict(self, imgs, return_acc=False):
        """
        TODO: describe method
        """
        Xs = []
        for img in imgs:
            Xs.append(self.normalize(img))

        predicted = [[], [], []]
        if bool(Xs):
            x = torch.tensor(np.moveaxis(np.array(Xs), 3, 1))
            if mode_torch == "gpu":
                x = x.cuda()
            predicted = self.MODEL(x)
            predicted = [p.cpu().numpy() for p in predicted]

        regionIds = []
        for region in predicted[0]:
            regionIds.append(int(np.argmax(region)))

        countLines = []
        for countL in predicted[1]:
            countLines.append(int(np.argmax(countL)))

        if return_acc:
            return regionIds, countLines, predicted
        
        return regionIds, countLines

    def getRegionLabel(self, index):
        """
        TODO: describe method
        """
        return self.CLASS_REGION[index].replace("-", "_")

    def getRegionLabels(self, indexes):
        """
        TODO: describe method
        """
        return [self.CLASS_REGION[index].replace("-", "_") for index in indexes]

    def compile_train_generator(self, train_dir, target_size, batch_size=32):
        """
        TODO: describe method
        """
        # with data augumentation
        imageGenerator = ImgGenerator(
            train_dir,
            self.WEIGHT,
            self.HEIGHT,
            self.BATCH_SIZE,
            [len(self.CLASS_REGION), len(self.CLASS_COUNT_LINE)])
        print("start train build")
        imageGenerator.build_data()
        print("end train build")
        return  imageGenerator

    def compile_test_generator(self, test_dir, target_size, batch_size=32):
        """
        TODO: describe method
        """
        imageGenerator = ImgGenerator(
            test_dir,
            self.WEIGHT,
            self.HEIGHT,
            self.BATCH_SIZE,
            [len(self.CLASS_REGION), len(self.CLASS_COUNT_LINE)])
        print("start test build")
        imageGenerator.build_data()
        print("end test build")
        return  imageGenerator