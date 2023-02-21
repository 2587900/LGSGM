from Controller_GCN import Trainer, Evaluator  #从Controller_GCN模块导入Trainer, Evaluator两个类
import torch    #PyTorch深度学习库，用于后续的模型训练和预测。
import os       #Python的标准库之一，用于处理操作系统相关的功能，如文件路径等。
import joblib   #Python的一个第三方库，用于高效地进行对象的序列化和反序列化操作。

"""
这段代码创建了一个字典 info_dict 并向其中添加了一个键值对。
键是 'save_dir'，值是字符串 './Report'。
这个字典可能被用于配置和传递到一个 Trainer 或 Evaluator 类的实例中，
以控制训练和评估的行为。其中 'save_dir' 键可能表示将训练/评估结果保存的目录。
"""
"""
具体地：
numb_sample 是一个整数，表示一个epoch中训练的样本数量，默认为None，即使用所有可用的训练数据进行训练。
numb_epoch 是一个整数，表示训练的epoch数量。
numb_gcn_layers 是一个整数，表示要叠加的GIN层数。
gcn_hidden_dim 是一个列表，表示每个GIN层中的隐藏层维度。
gcn_output_dim 是一个整数，表示图嵌入向量的维度。
gcn_input_dim 是一个整数，表示图的节点和边的维度。
batchnorm 是一个布尔值，表示是否在GCN的每个层后使用BatchNorm。
batch_size 是一个整数，表示批量大小。
dropout 是一个浮点数，表示dropout的概率。
visual_backbone 是一个字符串，表示用于提取视觉特征的EfficientNet的backbone。
visual_ft_dim 是一个整数，表示从EfficientNet中提取的视觉特征的维度。
optimizer 是一个字符串，表示使用的优化器，可以是'Adam'或'SGD'等。
learning_rate 是一个浮点数，表示学习率。
activate_fn 是一个字符串，表示使用的激活函数，可以是'swish'、'relu'或'leakyrelu'等。
grad_clip 是一个浮点数，表示梯度裁剪的阈值。
model_name 是一个字符串，表示要使用的模型名称。
checkpoint 是一个字符串，表示预训练模型的路径。
margin_matrix_loss 是一个浮点数，表示矩阵损失的边缘。
rnn_numb_layers 是一个整数，表示RNN的层数。
rnn_bidirectional 是一个布尔值，表示RNN是否是双向的。
rnn_structure 是一个字符串，表示使用的RNN结构，可以是'LSTM'或'GRU'。
graph_emb_dim 是一个整数，表示图嵌入向量的维度。
include_pred_ft 是一个布尔值，表示是否包括视觉谓词特征。
freeze 是一个布尔值，表示是否冻结模型的所有层，除了GCN和图嵌入模块。
"""
info_dict = dict()
info_dict['save_dir'] = './Report'
info_dict['numb_sample'] = None # training sample for 1 epoch
info_dict['numb_epoch'] = 100 # number of epoch
info_dict['numb_gcn_layers'] = 1 # number of gin layers to be stacked
info_dict['gcn_hidden_dim'] = [] # hidden layer in each gin layer
info_dict['gcn_output_dim'] = 1024 # graph embedding final dim
info_dict['gcn_input_dim'] = 2048 # node and edges dim of a graph
info_dict['batchnorm'] = True
info_dict['batch_size'] = 128
info_dict['dropout'] = 0.5
info_dict['visual_backbone'] = 'b5' # EfficientNet backbone to extract visual features
info_dict['visual_ft_dim'] = 2048
info_dict['optimizer'] = 'Adam' # or Adam
info_dict['learning_rate'] = 3e-4
info_dict['activate_fn'] = 'swish' # swish, relu, leakyrelu
info_dict['grad_clip'] = 2 # Gradient clipping
# info_dict['use_residual'] = False # always set it to false (not implemented yet)
# Embedder for each objects and predicates, embed graph only base on objects
info_dict['model_name'] = 'GCN_ObjAndPredShare_NoFtExModule_LSTM' 
info_dict['checkpoint'] = None # Training from a pretrained path
info_dict['margin_matrix_loss'] = 0.35
info_dict['rnn_numb_layers'] = 2
info_dict['rnn_bidirectional'] = True
info_dict['rnn_structure'] = 'LSTM' # LSTM or GRU (LSTM gave better result)
info_dict['graph_emb_dim'] = info_dict['gcn_output_dim']*2
info_dict['include_pred_ft'] = True # include visual predicate features or not
info_dict['freeze'] = False # Freeze all layers except the graph convolutional network and graph embedding module

def run_train(info_dict):
    """
    如果路径文件不存在
    则输出：将创建文件夹
    并创建一个文件夹，路径为 info_dict 参数中指定的文件夹
    """
    if not os.path.exists(info_dict['save_dir']):
        print(f"Creating {info_dict['save_dir']} folder")
        os.makedirs(info_dict['save_dir']) 
        
    trainer = Trainer(info_dict)    #创建一个名为 trainer 的深度学习模型训练器，它接受一个名为 info_dict 的参数
    trainer.train()     #调用 trainer 对象的 train 方法，开始训练模型。

    # 读取数据
    subset = 'test'
    DATA_DIR = './Data'
    images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
    caps_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
    
    """
    调用 trainer 对象的 validate_retrieval 方法进行模型评估。
    images_data 和 caps_data 分别是用于评估的图像和文本数据。
    False表示在评估时不要将图像和文本数据组合在一起。
    lossVal 是评估时计算得到的损失值。
    ar_val 和 ari_val 分别是计算出来的指标值。
    """
    lossVal, ar_val, ari_val = trainer.validate_retrieval(images_data, caps_data, False)
    info_txt = f"\n----- SUMMARY (Matrix)-----\nLoss Val: {6-lossVal}"   
    info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    print(info_txt)
    
    lossVal, ar_val, ari_val = trainer.validate_retrieval(images_data, caps_data, True)
    info_txt = f"\n----- SUMMARY (Combine)-----\nLoss Val: {6-lossVal}"   
    info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    print(info_txt)
    
def run_evaluate(info_dict):
    # path to pretrained model
    info_dict['checkpoint'] = './Report/GCN_ObjAndPredShare_NoFtExModule_LSTM_Freeze-16022021-030527.pth.tar'

    evaluator = Evaluator(info_dict)
    evaluator.load_trained_model()
    
    subset = 'test'
    DATA_DIR = './Data'
    images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
    caps_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
    
    #lossVal, ar_val, ari_val = evaluator.validate_retrieval(images_data, caps_data, False)
    #info_txt = f"\n----- SUMMARY (Matrix)-----\nLoss Val: {6-lossVal}"   
    #info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    #info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    #print(info_txt)
    
    lossVal, ar_val, ari_val = evaluator.validate_retrieval(images_data, caps_data, True)
    info_txt = f"\n----- SUMMARY (Combine)-----\nLoss Val: {6-lossVal}"   
    info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    print(info_txt)
    
run_train(info_dict)
run_evaluate(info_dict)
