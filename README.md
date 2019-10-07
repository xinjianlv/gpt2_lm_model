# GPT2-LM-MODEL

##Description 
**主要功能：**

使用GPT-2训练中文语言模型，目前只支持单GPU。采用ignite框架，代码较清晰，易理解。

**结构描述：**

	./config/  : 保存tokenizer及模型的配置文件
	./data/	   : 训练数据
	./cache/   : 缓存文件夹
    ./model/   : 保存模型的文件夹
    ./src/     : 源码文件夹
    run_tokenize.sh ： 预编码脚本
    run_train.sh ：训练脚本
##Usage
**源码下载：**
	git clone http:

**使用：**

1）使用demo数据：

	a)训练对话语料：直接运行run_train.sh脚本即可：
	--dialogue 值为1时，训练对话语料 
	--dataset_path 为数据路径选项，指定了原始数据，则会直接加载原始数据进行处理。
	b)训练小说语料：需要先运行run_tokenize.sh脚本对小说数据进行预处理。之后再运行run_train.sh脚本。
	--tokenized_data_path ：run_tokenize.sh脚本预处理后的文件夹路径。
	
2）模型使用：

	./src/interact.py
	--model_file 模型文件
	--vocab_file tokenizer使用的词表文件
	
	
