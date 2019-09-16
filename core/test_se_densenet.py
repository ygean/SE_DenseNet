import torch
import torch.utils.model_zoo as model_zoo
from se_densenet import se_densenet121
# from official repo import densenet
from torchvision.models.densenet import densenet121


def test_se_densenet(pretrained=False):
    X = torch.Tensor(32, 3, 224, 224)

    if pretrained:
        model = se_densenet121(pretrained=pretrained)
        net_state_dict = {key: value for key, value in model_zoo.load_url("https://download.pytorch.org/models/densenet121-a639ec97.pth").items()}
        model.load_state_dict(net_state_dict, strict=False)

    else:
        model = se_densenet121(pretrained=pretrained)

    # print(model)
    if torch.cuda.is_available():
        X = X.cuda()
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        output = model(X)
        print(output.shape)


def test_densenet():
    """create example tensor data for densenet, and print output variable shape"""
    X = torch.Tensor(32, 3, 224, 224)

    model = densenet121(pretrained=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()

    model.eval()
    with torch.no_grad():
        output = model(X)
        print(output.shape)


if __name__ == "__main__":
    test_se_densenet(pretrained=True)