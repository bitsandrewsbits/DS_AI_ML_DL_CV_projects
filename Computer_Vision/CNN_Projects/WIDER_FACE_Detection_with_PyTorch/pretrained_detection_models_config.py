# object-detection pre-trained models from torchvision
from torchvision.models import detection

PRETRAINED_DETECTION_MODELS = {
		"fcos_resnet50_fpn": {
			"model": detection.fcos_resnet50_fpn,
			"weights": detection.FCOS_ResNet50_FPN_Weights.DEFAULT
		},
		"fasterrcnn_mobilenet_v3_large_320_fpn": {
			"model": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
			"weights": detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
		},
		"fasterrcnn_mobilenet_v3_large_fpn": {
			"model": detection.fasterrcnn_mobilenet_v3_large_fpn,
			"weights": detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
		}
	}