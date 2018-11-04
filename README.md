# SE_DenseNet

## Introduction
![pic-0](assets/03.jpeg)
This is a DensNet  which contains a SE (Squeeze-and-Excitation Networks by Jie Hu, Li Shen and Gang Sun) module.

The backbone is densenet, I just add senet into densenet as pic shows, but it's not whole stucture of se_densenet, please check **[blog](http://www.zhouyuangan.cn/2018/11/%E5%88%A9%E7%94%A82017%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C%E5%86%A0%E5%86%9B%E7%BD%91%E7%BB%9Csqueeze-and-excitation-%E7%BD%91%E7%BB%9C%E4%BF%AE%E6%94%B9densenet/)** you will get more details on se_densenet, and then I will test se-densenet on my own classification task and dataset compares with performance of densenet.

![pic-1](assets/02.png)

 please click my **[blog](http://www.zhouyuangan.cn/2018/11/%E5%88%A9%E7%94%A82017%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C%E5%86%A0%E5%86%9B%E7%BD%91%E7%BB%9Csqueeze-and-excitation-%E7%BD%91%E7%BB%9C%E4%BF%AE%E6%94%B9densenet/)**  if you want to kown more edited se_densenet details.

## Usage

For test se_densenet
```
python se_densenet.py
```
And it will print structure of se_densenet.

Let's input an tensor which shape is (32, 3, 224, 224) into se_densenet

```
python test_se_densenet.py
```
 Of course, it will print ``torch.size(32, 1000)``

## Test and result

### densenet

- train
![](assets/densenet121_train_acc.png)
![](assets/densenet121_train_loss.png)

- val
![](assets/densenet121_val_acc.png)
![](assets/densenet121_val_loss.png)

the best acc is: 98.5417%

### se_densenet

- train
![](assets/se_densenet121_train_acc.png)
![](assets/se_densenet121_train_loss.png)

-val
![](assets/se_densenet121_val_acc.png)
![](assets/se_densenet121_val_loss.png)

the best acc is: 98.6154%

### tabel

||best train acc|best val acc|
|--|--|--|
|densenet|0.966953|0.985417|
|se_densenet|**0.967772**|**0.986154**|

Se_densenet has got **0.0737%** higher accuracy than densenet. I don't train and test on public dataset like cifar and coco, because of low capacity of machine computation, you can train and test on cifar or coco dataset by yourself if you have the will.

## TODO

I will update content and show my test result as quickly as possible.

- [x] usage of my codes
- [x] test result on my own dataset
- [ ] train and test public dataset, if I have enough time