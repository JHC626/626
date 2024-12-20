# Jetson AGX Orin 加装固态踩的坑
1.首先就是一定要插严SSD否则可能在lsblk中无法识别
2.进行分区开始
```bash
sudo fdisk /dev/nvme0n1
```
注：这里按p可以查看分区情况，即便系统显示你没有分区，这时也有可能会出现显示分区空间不足的情况，这时应该去找到桌面上的disk然后点击右上角的 format disk 格式化磁盘，重新进行分区操作。n为新建分区，区域的起止地址默认即可，按w退出并保存。
3.分区建立成功后，这里注意不要进行重启，重启会造成分区失败！
4.进入到root用户后
```bash
sudo su -
```
```bash
mkfs.ext4 /dev/nvme0n1p1
```
```bash
mkdir /home1
```
```bash
mount /dev/nvme0n1p1 /home1
```
此时 df -h 就可以看见ssd的存储空间了

给予sirs用户权限
```bash
chown -R sirs:sirs /home1
```

5.设置自动开机挂载，这里一定要注意，继续在root下操作，否则更改了也是无效的
``` bash
vi /etc/fstab
```
按下insert键进入编辑模式，编辑后再按一次insert保存编辑内容，编辑完成后按下esc后输入 ：wq！ 后再回车即可退出。       
6.修改conda环境保存路径
```bash
chmod 777 /home1/miniconda3
vim ~/.condrac
```
 打开后加入
```bash
envs_dirs:
   - /home1/miniconda3/my_envs
pkgs_dirs:
   - /home1/miniconda3/my_pkgs
```

# Jetson AGX Orin 更新jetpack踩的坑
1.首先如果想更新 jetpack 需要一台原生 ubuntu 系统电脑，虚拟机是不可以的，注意jetson的电源线和连接主机的数据线虽然都是 type-c 接口，但是二者不能接反，数据线接在排针的位置，电源线和显示信号输出接口在一侧，否则sdk软件无法识别。
2.在进行第三步安装时会报错
(1)![2676c908f8099bea806cc2f862a0e4c](https://github.com/user-attachments/assets/ceeb653c-12fd-40ee-9e4a-4f3ef0e3439d)
此处报错可以删除寒武纪相关安装包来解除
```bash
sudo apt-get purge cambricon-mlu-driver-ubuntu20.04-dkms

```
(2)QGIS相关的报错
```bash
sudo nano /etc/apt/sources.list.d/qgis.sources
```
打开后将$your_distributions_codename替换为“jammy”,类似的错误也可能发生在其他的文件中例如

(3)
```bash
sudo nano /etc/apt/sources.list.d/docker.list
```
如果该文件包含 https://download.docker.com/linux/ubuntu， 将其替换为 https://mirrors.aliyun.com/docker-ce/linux/ubuntu/

(4)并且第三条docker的问题实际上并不确定此方法能否实际解决一些报错，在进入第三步安装时，docker的安装也就是 nvidia runtime 实际上还会安装失败，最好的解决办法就是连接外网进行下载，由于实验室条件限制，可以通过局域网代理来解决，通过笔记本电脑上的v2ray软件来实现。

a.打开 ubuntu 设备的设置->network。

b.找到 network proxy 第一列的四行全部输入笔记本电脑的ip地址

c.第二列的四行按照v2ray左下角的数字填写http和socks
![image](https://github.com/user-attachments/assets/36609dc6-99a7-4def-98bb-a65adca4516c)
这样连接外网后，docker下载就不会有问题了,想使用 jetson 连接外网使用同样的方法进行代理即可。

(5)如果设备连接好后，sdk软件检测不到设备可如此解决
![image](https://github.com/user-attachments/assets/42113d0a-929a-45f7-a818-b41cc2d3cc16) 
（转载自csdn博主 抢公主的大魔王）

详细见https://blog.csdn.net/weixin_43111445/article/details/135446335
注：查看设备的ip时一般用ifconfig 不是hostname -i的那个ip，还要注意连接在同一局域网下
# 使用VNC远程操作jetson
可完全参考https://bbs.huaweicloud.com/blogs/350523 张辉博主的讲解一部一部操作即可，如果遇到connection refused可使用
```bash
/usr/lib/vino/vino-server --display=:0
```
即可解决
# 更新jetpack后的硬盘分区
当更新jetpack后会发现，之前的硬盘虽然分区还在，但是挂载的文件夹出现了错误，重新挂载即可
```bash
mkdir /JHC
```
```bash
mount /dev/nvme0n1p1 /JHC
```
# jetson一些库和环境的的安装
1.jetson由于是arm架构，只能使用Chromium
```bash
sudo apt-get update
sudo apt-get install chromium-browser
```
```bash
sudo apt update
```
就会下载成功，此处确实莫名其妙

2.安装conda环境

登录miniconda官网下载安装包后直接安装就可以，一路enter和yes就可以，最后如果不想选择默认路径，如我想保存在JHC/miniconda3目录下，我现在有JHC文件夹，则我不应手动创建 miniconda3 文件夹，我只要输入路径JHC/miniconda3即可，否则它会提示你已存在文件夹JHC/miniconda3，报错。

3.在本机安装好tensorrt还需要复制到conda环境中才能在conda中使用，正常装jetpack时已经包括了 tensorrt 等工具包在本机中
```bash
export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* JHC/miniconda3/envs/{您的虚拟环境名字}/lib/python${PYTHON_VERSION}/site-packages/
```
4.安装pytorch以及torchvision

https://pytorch.org/TensorRT/getting_started/jetpack.html
此官网提供的确实是jetpack6.1的适应pytorch版本，torch==2.5.0，但是torchvision会出现问题

12.5日更新：如果按照官网的版本安装 torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl ，print(torch.cuda.is_available()) ，print(torch.cuda.device_count())这两个的输出是没有问题的，但是当安装配套的torchvision==0.20.0时就会出现报错，无法导入torchvision，实际上就是这一代版本的问题，如果将 torch 升级到2.5.1，torchvision到0.20.1
```bash
pip install --upgrade torch torchvision
```
print(torch.cuda.is_available()) ，print(torch.cuda.device_count())，会出现无法使用cuda及gpu的问题，说明torch的版本不匹配，此时我们应该将torch和torchvision的版本同时降级到jetpack==6.0，cuda==12.4时的版本可以解决这个问题，但是numpy版本要安装1.x版本，推荐numpy=1.21.0

此为jp==6.0，cuda==12.4的下载网址。
http://jetson.webredirect.org/jp6/cu124

5.jtop无法显示
```bash
export TERM='xterm-256color'
```
