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

## TODO

I will update content and show my test result as quickly as possible.

- [x] usage of my codes
- [ ] test result