import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torch.nn import init
from PIL import Image
import numpy as np
import cv2.cv2 as cv

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64

        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #注意力CBAM
        self.CBAM0 = CBAMBlock(channel=self.inplanes,kernel_size=7)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 注意力CBAM
        self.CBAM1 = CBAMBlock(channel=self.inplanes, kernel_size=7)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.CBAM0(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.CBAM1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def validate(valid_data,model):
    with torch.no_grad():
        valid_acc = 0.0
        for j, (input, label) in enumerate(valid_data):
            input = input.to(device)
            label = label.to(device)

            output = model.forward(input)
            output.cuda()

            ret, prediction = torch.max(output.data, 1)

            correct_counts = prediction.eq(label.data.view_as(prediction))
            acc_val = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_acc += acc_val.item() * input.size(0)
    return valid_acc


'''
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier'''


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual



if __name__=='__main__':
    model=ResNet(Bottleneck,[3,4,6,3])
    # data_dir_angry="E:/图片数据集/数据集/ImageNet/ImageNet/angry"
    # data_dir_happy="E:/图片数据集/数据集/ImageNet/ImageNet/happy"
    # data_dir_sad="E:/图片数据集/数据集/ImageNet/ImageNet/sad"
    # data_dir_sleepy="E:/图片数据集/数据集/ImageNet/ImageNet/sleepy"
    # data_dir_angry_valid = "E:/图片数据集/数据集/ImageNet/ImageNet/angry_valid"
    # data_dir_happy_valid = "E:/图片数据集/数据集/ImageNet/ImageNet/happy_valid"
    # data_dir_sad_valid = "E:/图片数据集/数据集/ImageNet/ImageNet/sad_valid"
    # data_dir_sleepy_valid = "E:/图片数据集/数据集/ImageNet/ImageNet/sleepy_valid"
    data_dir_train="E:\PY\PyProjects39\project1\Code\DeepLearning\ResNet\splitDataset\\train"
    data_dir_valid = "E:\PY\PyProjects39\project1\Code\DeepLearning\ResNet\splitDataset\\val"

    # filenames_angry= [name for name in os.listdir(data_dir_angry) if name.endswith('.jpg') ]
    # filenames_happy = [name for name in os.listdir(data_dir_happy) if name.endswith('.jpg')]
    # filenames_sad = [name for name in os.listdir(data_dir_sad) if name.endswith('.jpg')]
    # filenames_sleepy = [name for name in os.listdir(data_dir_sleepy) if name.endswith('.jpg')]
    # filenames_angry_valid = [name for name in os.listdir(data_dir_angry_valid) if name.endswith('.jpg')]
    # filenames_happy_valid = [name for name in os.listdir(data_dir_happy_valid) if name.endswith('.jpg')]
    # filenames_sad_valid = [name for name in os.listdir(data_dir_sad_valid) if name.endswith('.jpg')]
    # filenames_sleepy_valid = [name for name in os.listdir(data_dir_sleepy_valid) if name.endswith('.jpg')]

    # filename_train=[name for name in os.listdir(data_dir_train) if name.endswith('.jpg')]
    # filename_valid= [name for name in os.listdir(data_dir_valid) if name.endswith('.jpg')]

    transform_train=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_valid = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_train=datasets.ImageFolder(root=data_dir_train,transform=transform_train)
    data_valid= datasets.ImageFolder(root=data_dir_valid, transform=transform_valid)

    train_data=DataLoader(data_train,batch_size=12,shuffle=True)
    valid_data=DataLoader(data_valid,batch_size=12,shuffle=True)
    # data_train_angry=datasets.ImageFolder(root=data_dir_angry,transform=transform_train)
    # data_valid_angry = datasets.ImageFolder(root=data_dir_angry_valid, transform=transform_valid)
    #
    # train_data_angry=DataLoader(data_train_angry,batch_size=32,shuffle=True)
    # valid_data_angry=DataLoader(data_valid_angry,batch_size=32,shuffle=True)
    #
    # data_train_happy=datasets.ImageFolder(root=data_dir_happy, transform=transform_train)
    # data_valid_happy = datasets.ImageFolder(root=data_dir_happy_valid, transform=transform_valid)
    # train_data_happy = DataLoader(data_train_happy, batch_size=32, shuffle=True)
    # valid_data_happy = DataLoader(data_valid_happy, batch_size=32, shuffle=True)
    #
    # data_train_sad = datasets.ImageFolder(root=data_dir_sad, transform=transform_train)
    # data_valid_sad = datasets.ImageFolder(root=data_dir_sad_valid, transform=transform_valid)
    # train_data_sad = DataLoader(data_train_sad, batch_size=32, shuffle=True)
    # valid_data_sad = DataLoader(data_valid_sad, batch_size=32, shuffle=True)
    #
    # data_train_sleepy = datasets.ImageFolder(root=data_dir_sleepy, transform=transform_train)
    # data_valid_sleepy = datasets.ImageFolder(root=data_dir_sleepy_valid, transform=transform_valid)
    # train_data_sleepy = DataLoader(data_train_sleepy, batch_size=32, shuffle=True)
    # valid_data_sleepy = DataLoader(data_valid_sleepy, batch_size=32, shuffle=True)




    loss_fun=nn.CrossEntropyLoss()
    loss_fun.cuda()
    optimizer=optim.Adam(model.parameters(),lr=0.01)

    device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu" )
    print(device)
    # model.to(device)
    model.cuda()
    epochs=30
    history=[]
    best_acc=0.0
    best_epoch=0
    ls_train=[]
    ls_val=[]
    for epoch in range(epochs):
        epoch_start=time.time()
        print("Epoch{}/{}".format(epoch+1,epochs))
        model.train()
        train_loss=0.0
        train_acc=0.0
        valid_loss=0.0


        for i ,(inputs,labels)in enumerate(train_data):

            inputs=inputs.to(device)
            inputs.cuda()
            labels=labels.to(device)
            labels.cuda()
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            outputs.cuda()
            loss=loss_fun(outputs,labels)

            loss.backward()
            optimizer.step()


            train_loss+=loss.item()*inputs.size(0)
            ret,predictions=torch.max(outputs.data,1)
            correct_counts=predictions.eq(labels.data.view_as(predictions))
            acc_train=torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc+=acc_train.item()*inputs.size(0)

        valid_acc=validate(valid_data, model)

        avg_train_loss=train_loss/len(data_train)
        avg_train_acc = train_acc / len(data_train)

        avg_valid_loss = valid_loss / len(data_valid)
        avg_valid_acc = valid_acc / len(data_valid)
        history.append([avg_train_acc,avg_train_loss,avg_valid_acc,avg_valid_loss])
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        if best_acc<avg_valid_acc:
            best_acc=avg_valid_acc
            best_epoch=epoch+1
        epoch_end = time.time()
        print(
                    "Epoch: {:03d},Training Loss: {:.4f},Accuracy: {:.4f}%\n\t\tAccuracy: {:.4f}%,Time: {:.4f}s".format(
                        epoch + 1, avg_train_loss, avg_train_acc * 100,  avg_valid_acc * 100,
                        epoch_end - epoch_start))
        ls_train.append(avg_train_acc)
        ls_val.append(avg_valid_acc)

    print("Best Accuracy for validation:{:.4f} at epoch {:03d}".format(best_acc,best_epoch))
    torch.save(model.state_dict(),'E:\PY\PyProjects39\project1\Code\DeepLearning\ResNet'+str(epoch+1)+'.pt')

    #训练曲线可视化
    ep = [i for i in range(1,epochs+1)]
    fg = plt.figure()
    ax = fg.add_subplot(1, 1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.plot(ep, ls_train, "-r", label="training accuracy")
    s1 = plt.scatter(ep, ls_train, color='r')
    ax2_share_x = ax.twinx()
    ax2_share_x.set_ylabel("valid accuracy")
    ax2_share_x.plot(ep, ls_val, '-b', label='valid accuracy')
    s2 = plt.scatter(ep, ls_val, color='b')
    plt.legend((s1, s2), ('training accuracy', 'valid accuracy'))

    plt.title("epoch-accuracy")
    plt.show()


    #注意力热力图可视化
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    target_layers = [model.layer4[-1]]
    image_size=224
    img = Image.open('E:\PY\PyProjects39\project1\Code\DeepLearning\ResNet\splitDataset\\train\\angry\\0angry.jpg').convert('RGB')
    img = np.array(img, dtype=np.uint8)  # 转成np格式
    img = cv.resize(img, (image_size,image_size))  # 将测试图像裁剪成跟训练图片尺寸相同大小的
    img_array = np.array(img)
    img_tensor = data_transform(img_array)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('./result.png')  # 将热力图的结果保存到本地当前文件夹
    plt.show()

    img = Image.open('E:\PY\PyProjects39\project1\Code\DeepLearning\ResNet\splitDataset\\train\sleepy\\91sleepy.jpg').convert('RGB')
    img = np.array(img, dtype=np.uint8)  # 转成np格式
    img = cv.resize(img, (image_size, image_size))  # 将测试图像裁剪成跟训练图片尺寸相同大小的
    img_array = np.array(img)
    img_tensor = data_transform(img_array)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('./result1.png')  # 将热力图的结果保存到本地当前文件夹
    plt.show()