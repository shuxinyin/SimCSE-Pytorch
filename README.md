## SimCSE Inplemention

SimCSE在中文上无监督 + 有监督 pytorch版
> SimCSE：https://arxiv.org/pdf/2104.08821.pdf   
> ESimCSE: https://arxiv.org/pdf/2109.04380.pdf

1.database: SNS-B (uploaded)
> directory: data/SNS-B/  

2.environment
> torch==1.8.2  
> transformers==4.12.3
> 
> video card: 3060Ti 8G    
> Due to the limitation of the graphics card, the batch_size is set very small.  
> You can try increasing the batch_size to get better results with video memory allowed.  

3.how to run?
> SimCSE： python train.py  
> ESimCSE： python ESimCSE_train.py

4.Result (un-supervised)  
**spearman corrcoef** is shown as result below:

| Model     | un_supervised |
|-----------|---------------|
| Bert_base | 0.538         |   
| SimCSE    | 0.692         |
| ESimCSE   | 0.707         | 

说明：原论文的无监督SimCSE基于英文，从维基百科上挑了100万个句子进行训练的。本项目评测实验是在中文数据集STS-B(已上传)，实现结果以[苏剑林科学空间结果](https://spaces.ac.cn/archives/8348) 对照。
SimCSE结果与其一致。   
以上供参考， 码代码不易，有用请点赞。  
![img.png](./data/pic/img.png)  
