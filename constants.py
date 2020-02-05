# Model constants
DEEPLAB = 'deeplab'
DEEPLAB_50 = 'deeplab_50'
DEEPLAB_34 = 'deeplab_34'
DEEPLAB_18 = 'deeplab_18'
DEEPLAB_MOBILENET = 'deeplab_mn'
DEEPLAB_MOBILENET_DILATION = 'deeplab_mnd'
UNET = 'unet'
UNET_PAPER = 'unet_paper'
UNET_PYTORCH = 'unet_torch'
PSPNET = 'pspnet'

# Dataset constants
DOCUNET = 'docunet'
DOCUNET_INVERTED = 'docunet_inverted'
DOCUNET_IM2IM = 'docunet_im2im'

# Dataset locations
HAZMAT_DATASET = 'hazmat_dataset'
ADDRESS_DATASET = 'address_dataset'
LABELS_DATASET = 'labels_dataset'

# loss constants
DOCUNET_LOSS = 'docunet_loss'
MS_SSIM_LOSS = 'ms_ssim_loss'
MS_SSIM_LOSS_V2 = 'ms_ssim_loss_v2'
SSIM_LOSS = 'ssim_loss'
SSIM_LOSS_V2 = 'ssim_loss_v2'
SMOOTH_L1_LOSS = 'smoothl1_loss'
L1_LOSS = 'l1_loss'
MSE_LOSS = 'mse_loss'

# Optimizers
SGD = 'sgd'
AMSGRAD = 'amsgrad'
ADAM = 'adam'
RMSPROP = 'rmsprop'
ADABOUND = 'adabound'

# Normalization layers
INSTANCE_NORM = 'instance'
BATCH_NORM = 'batch'
SYNC_BATCH_NORM = 'syncbn'

# Init types
NORMAL_INIT = 'normal'
KAIMING_INIT = 'kaiming'
XAVIER_INIT = 'xavier'
ORTHOGONAL_INIT = 'orthogonal'

# Downsampling methods
MAXPOOL = 'maxpool'
STRIDECONV = 'strided'

# Split constants
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
TRAINVAL = 'trainval'
VISUALIZATION = 'visualization'