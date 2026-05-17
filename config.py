CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

CXR_IRGEN_CLASSES = [
    'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
    'enlarged cardiomediastinum', 'fracture', 'lung lesion', 'lung opacity',
    'no finding', 'pleural effusion', 'pleural other', 'pneumonia',
    'pneumothorax', 'support devices',
]

COMPETITION_5 = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 14
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TR_MAX_EPOCH = 3
IMG_RESIZE = 320
IMG_CROP = 224
FILTER_THRESHOLD = 0.7

WP1_MACRO_AUC = 0.8782
WP3_MACRO_AUC = 0.8650
WP4_MACRO_AUC = 0.8742
