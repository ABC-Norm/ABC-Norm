from torchvision import transforms

# Data transformation with augmentation
class augmentation:
    def __init__(self, resize=256, size=224, is_train=False):
        self.resize = resize
        self.size = size
        self.is_train = is_train

        self.augment = self.build()

    def train_aug(self):
        augment = transforms.Compose([
                  # transforms.RandomResizedCrop(self.size, scale=(0.4, 1.0)),
                  transforms.RandomResizedCrop(self.size),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(degrees=20),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
        return augment

    def valid_aug(self):
        augment = transforms.Compose([
                  transforms.Resize(self.resize),
                  transforms.CenterCrop(self.size),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
        return augment

    def build(self):
        if self.is_train:
            return self.train_aug()
        else:
            return self.valid_aug()

    def __call__(self, x):
        return self.augment(x)

