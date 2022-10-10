
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode
        self.input_size = (1, 80, 80, 32)
        self.checkpoint_dir = "checkpoint/"

        if self.mode == PRETRAINING:
            self.batch_size = 8
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 50
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 1 # depends on the meta-data at hand
            self.temperature = 0.1
            self.model = "DenseNet"

            #self.pretrained_path = "DenseNet121_BHB-10K_yAwareContrastive.pth"
            #self.pretrained_path = "checkpoint/ntxent_Contrastive_MRI_epoch_21_80_80_32.pth"
            #self.pretrained_path = "checkpoint/ntxent_Contrastive_MRI_epoch_20.pth"

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 10
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            #self.pretrained_path = "checkpoint/ntxent_Contrastive_MRI_epoch_50_80_80_32_mam.pth"
            self.pretrained_path = "checkpoint/ntxent_Contrastive_MRI_epoch_12.pth"
            self.num_classes = 5
            self.model = "DenseNet"
