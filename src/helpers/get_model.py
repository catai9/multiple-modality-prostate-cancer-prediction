import torch
import monai


def get_model(device, path_to_weights, use_pretrained_sbr_model=False):
    if use_pretrained_sbr_model:
        print("Using pretrained sbr model")
        model = get_trained_model(
            path_to_weights
        )
    else:
        model = fix_base_resnet_weights(device, path_to_weights)
    return model.to(device)


def fix_base_resnet_weights(device, path_to_weights):
    model = monai.networks.nets.resnet34(
        spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=False
    )

    model_dict = model.state_dict()

    pretrain = torch.load(path_to_weights)
    pretrain["state_dict"] = {
        k.replace("module.", ""): v for k, v in pretrain["state_dict"].items()
    }

    pretrain["state_dict"] = {
        k: v for k, v in pretrain["state_dict"].items() if k in model_dict.keys()
    }
    model.load_state_dict(pretrain["state_dict"], strict=False)

    model.conv1 = torch.nn.Conv3d(
        3, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False
    )

    model = model.to(device)

    return model


def get_trained_model(path_to_weights):
    model = monai.networks.nets.resnet34(
        spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=False
    )

    model.conv1 = torch.nn.Conv3d(
        3, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False
    )

    model.load_state_dict(torch.load(path_to_weights, map_location="cpu"))

    return model

