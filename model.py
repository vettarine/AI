import timm

def get_model(num_classes, pretrained=True):
    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model
