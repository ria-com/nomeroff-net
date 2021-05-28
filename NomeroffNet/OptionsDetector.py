import os
import sys
from typing import List, Dict, Tuple
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))
from mcm.mcm import (download_latest_model,
                     download_model)
from mcm.mcm import get_mode_torch
from nnmodels import NPOptionsNet
from ImgGenerator import ImgGenerator

mode_torch = get_mode_torch()

CLASS_REGION_ALL = [
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
            "kg",
            "am"
        ]


def imshow(img: np.ndarray) -> None:
    """
    # functions to show an image
    """
    import matplotlib.pyplot as plt
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class OptionsDetector(ImgGenerator):
    """
    TODO: describe class
    """
    def __init__(self, options: Dict = None) -> None:
        """
        TODO: describe __init__
        """
        if options is None:
            options = dict()

        # input
        self.HEIGHT = 64
        self.WEIGHT = 295
        self.COLOR_CHANNELS = 3

        # outputs 1
        self.CLASS_REGION = options.get("class_region", CLASS_REGION_ALL)

        # outputs 2
        self.COUNT_LINES = options.get("count_lines", [
            0,
            1,
            2,
            3
        ])

        # model
        self.MODEL = None

        # train hyperparameters
        self.BATCH_SIZE = 64
        self.EPOCHS = 100

        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    @staticmethod
    def get_class_region_all() -> List:
        return CLASS_REGION_ALL

    def create_model(self) -> NPOptionsNet:
        """
        TODO: describe method
        """
        self.MODEL = NPOptionsNet(len(self.CLASS_REGION), len(self.COUNT_LINES), self.HEIGHT, self.WEIGHT)
        if mode_torch == "gpu":
            self.MODEL = self.MODEL.cuda()
        return self.MODEL

    def prepare(self, base_dir: str, verbose: bool = True) -> None:
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
        self.train_generator = self.compile_train_generator(train_dir)
        self.validation_generator = self.compile_test_generator(validation_dir)
        self.test_generator = self.compile_test_generator(test_dir)

        if verbose:
            print("DATA PREPARED")

    def train(self, log_dir: str = "./log", load_model: str = None, with_aug: bool = False) -> NPOptionsNet:
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
                inputs = torch.from_numpy(inputs)
                if mode_torch == "gpu":
                    inputs = inputs.cuda()
                outputs = self.MODEL(inputs)
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
                if i % diplay_per == 0:
                    g_loss /= diplay_per
                    g_loss_reg /= diplay_per
                    g_loss_line /= diplay_per
                    g_acc /= diplay_per
                    g_acc_reg /= diplay_per
                    g_acc_line /= diplay_per
                    
                    train_bar.set_description(f'[TRAIN {epoch + 1}, {i + 1}] '
                                              f'loss: {g_loss} '
                                              f'loss_reg: {g_loss_reg} '
                                              f'loss_line: {g_loss_line} '
                                              f'acc: {g_acc} '
                                              f'acc_reg: {g_acc_reg} '
                                              f'acc_line: {g_acc_line}')
                    g_loss = 0.0
                    g_loss_reg = 0.0
                    g_loss_line = 0.0
                    g_acc = 0
                    g_acc_reg = 0
                    g_acc_line = 0
            # validation
            val_acc, val_acc_reg, val_acc_line = self.test(test_generator=validationGenerator)
            print(f'[VALIDATION {epoch + 1}]',
                  f'val_acc: {val_acc} '
                  f'val_acc_reg: {val_acc_reg} '
                  f'val_acc_line: {val_acc_line} ')
            # save after each epochs
            model_path = os.path.join(
                log_dir,
                f"best.pb"
            )
            if val_acc > best_val_acc:
                torch.save(self.MODEL.state_dict(), model_path)
        print('Finished Training')
        return self.MODEL

    def test(self, test_generator: ImgGenerator = None) -> Tuple:
        """
        TODO: describe method
        """
        if test_generator is None:
            test_generator = self.test_generator.generator()
        all_acc = 0
        all_acc_reg = 0
        all_acc_line = 0
        n = 0
        for i, data in enumerate(test_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # test
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

    def save(self, path: str, verbose: bool = True) -> None:
        """
        TODO: describe method
        """
        if self.MODEL is not None:
            if bool(verbose):
                print("model save to {}".format(path))
            torch.save(self.MODEL.state_dict(), path)

    def isLoaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.MODEL is None:
            return False
        return True

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.create_model()
        if path_to_model == "latest":
            model_info = download_latest_model(self.get_classname(), "simple", mode=mode_torch)
            path_to_model = model_info["path"]
            options["class_region"] = model_info["class_region"]
        elif path_to_model.startswith("http"):
            model_info = download_model(path_to_model, self.get_classname(), "custom")
            path_to_model = model_info["path"]

        self.CLASS_REGION = options.get("class_region", CLASS_REGION_ALL)

        if mode_torch == "gpu":
            self.MODEL.load_state_dict(torch.load(path_to_model))
        else:
            self.MODEL.load_state_dict(torch.load(path_to_model,  map_location=torch.device('cpu')))
        self.MODEL.eval()
    
    @torch.no_grad()
    def predict(self, imgs: List[np.ndarray], return_acc: bool = False) -> Tuple:
        """
        TODO: describe method
        """
        Xs = []
        for img in imgs:
            Xs.append(self.normalize(img))

        predicted = [[], []]
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

    def getRegionLabel(self, index: int) -> str:
        """
        TODO: describe method
        """
        return self.CLASS_REGION[index].replace("-", "_")

    def getRegionLabels(self, indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [self.CLASS_REGION[index].replace("-", "_") for index in indexes]

    def compile_train_generator(self, train_dir: str) -> ImgGenerator:
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
        return imageGenerator

    def compile_test_generator(self, test_dir: str) -> ImgGenerator:
        """
        TODO: describe method
        """
        image_generator = ImgGenerator(
            test_dir,
            self.WEIGHT,
            self.HEIGHT,
            self.BATCH_SIZE,
            [len(self.CLASS_REGION), len(self.CLASS_COUNT_LINE)])
        print("start test build")
        image_generator.build_data()
        print("end test build")
        return image_generator
