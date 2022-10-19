#CLASSES = ['Bovidae', 'Cervidae', 'Canidae', 'Felidae', 'Mustelidae']
#CLASSES = ['Bovidae', 'Cervidae', 'Canidae', 'Felidae', 'Mustelidae', 'Giraffidae']
CLASSES = ['Bovidae', 'Cervidae', 'Canidae', 'Felidae', 'Mustelidae', 'Giraffidae', 'Pteropodidae', 'Procaviidae', 'Cercopithecidae', 'Delphinidae', 'Hyaenidae', 'Ursidae']
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode
        self.input_size = (1, 80, 80, 32)
        self.checkpoint_dir = "checkpoint/"

        if self.mode == PRETRAINING:
            #self.batch_size = 10
            self.batch_size = 32
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            self.temperature = 0.1
            self.model = "DenseNet"
            #self.loss = "NTXent"
            self.loss = "SupCon"

            #self.pretrained_path = "DenseNet121_BHB-10K_yAwareContrastive.pth"
            self.pretrained_path = "checkpoint/NTXent_epoch_45_OpenNeuro_with_BHB10K.pth"

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

            #self.pretrained_path = "checkpoint/NTXent_epoch_45_OpenNeuro_with_BHB10K.pth" 
            #self.pretrained_path = "checkpoint/SupCon_epoch_100_MaMI.pth"  
            #self.pretrained_path = "checkpoint/SupCon_epoch_52_all.pth"  
            self.pretrained_path = "checkpoint/fine_tune_epoch_100_all.pth"  
            #self.pretrained_path = "checkpoint/NTXent_epoch_86_MaMI.pth"  
            #self.pretrained_path = "checkpoint/NTXent_epoch_89_MaMI_bs32.pth"  

            #self.num_classes = 5
            self.num_classes = 12
            self.model = "DenseNet"
