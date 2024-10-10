# ComfyUI_Yccc
## 模型安装
模型会自动下载安装到本地的/ComfyUI/models/LLM/Phi-3.5-vision-instruct路径，但是需要提前挂好梯子。
## 节点功能
除了可以实现常规的单图单轮反推外，还可以实现单图多轮、以及多图单轮反推。另外，Phi模型对prompt的理解更好，以下是一些示例：
### 1.单图单轮
可以看到prompt是只对图片的背景环境做描述，而不描述人物进行反推，反推的结果贴合Prompt。
![3141728287422_ pic](https://github.com/user-attachments/assets/fbaca969-f5fd-459d-b0e2-edeec6d87567)
### 2.单图多轮
不同prompt之间用回车键隔开，即可切换为单图多轮。
![3191728355450_ pic](https://github.com/user-attachments/assets/cce9a54a-3c4b-426e-97c6-de52f92ea5ad)
### 3.多图单轮
将多图组成一个batch，即可切换为多图单轮。
![3131728285170_ pic](https://github.com/user-attachments/assets/b189b864-a561-4c0e-8d97-83551703ea65)
