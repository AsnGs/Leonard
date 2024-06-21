import operator
from collections import defaultdict
import pickle
import sys
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-param_file', action='store', dest='param_file',
                    help='param file file')
parser.add_argument('-output_path', action='store', dest='output_path',
                    help='input file path')
parser.add_argument('-input_path', action='store', dest='input_path',
                    help='input file path')
parser.add_argument('-edge_file', action='store', dest='edge_file',
                    help='input file path')     
parser.add_argument('-input_path1', action='store', dest='input_path1',
                    help='input file path')
# 模拟命令行输入
args = parser.parse_args([
    '-edge_file', './leonard/data/edges200m.npy',
    '-input_path', './leonard/raw_data/vertex200m.csv',
    '-input_path1', './leonard/raw_data/edge200m.csv',
    '-output', './leonard/data/vertex200m.npy',
    '-param', './leonard/data/vertex200m.params.json'
])

# 转换时间
def translate_time(time_):
    day=time_.split('.')
    order=int(day[1])
    day=int(day[0])
    return [day, order]

# 获取所有的 key:value 值, 同时保存特定type 的最小值
def get_dict_allkeys_values(dict_a,values,mins):
        for x in range(len(dict_a)):
            temp_key = list(dict_a.keys())[x]
            temp_value = dict_a[temp_key]
            if temp_key=='startTimestampNanos':
                if int(dict_a[temp_key])<=mins[0]:
                    mins[0]=int(dict_a[temp_key])
            elif temp_key=='time':
                tt, order=translate_time(dict_a[temp_key])
                if tt<=mins[2]:
                    mins[2]=tt
                if order<=mins[3]:
                    mins[3]=order
            elif temp_key=='childVertexHash' or temp_key=='parentVertexHash':
                continue
            else:
                if temp_key=='event id':
                    if int(dict_a[temp_key])<=mins[1]:
                        mins[1]=int(dict_a[temp_key])
                if temp_key not in values.keys():
                    values[temp_key]={}
                if str(temp_value) not in values[temp_key].keys():
                    values[temp_key][str(temp_value)]=len(values[temp_key].keys())
        return values
from collections import Counter
import re
import time
import sys

id2char_dict={}
char2id_dict={}
import csv
import copy
import numpy as np
def count():
    reader=[]
    data=[]
    recordd={}
    with open(args.input_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            data.append(i)
    key=data[0]
    data=data[1:]
    mins=[sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize]
    values={}
    data_tmp=[]
    counters_list=[]
    for i in range(len(data)):
        tmpdata=data[i]
        json_obj={}
        for j in range(len(key)):
            if tmpdata[j]!='':
                json_obj[key[j]]=tmpdata[j]
        values=get_dict_allkeys_values(json_obj,values,mins)  # 保存不同类型值的具体不同值对应索引的字典，{type:{value:index}}

    reader=[]
    data=[]
    print('end_vertex')
    with open(args.input_path1, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            data.append(i)
    key=data[0]
    data=data[1:]
    for i in range(len(data)):
        json_obj={}
        tmpdata=data[i]
        for j in range(len(key)-1):
            if tmpdata[j+1]!='':
                json_obj[key[j+1]]=tmpdata[j+1]
        values=get_dict_allkeys_values(json_obj,values,mins)
    # for i in values.keys():
    #     tmpset=values[i]
    #     tmp_dict={}
    #     begin=0
    #     for j in tmpset:
    #         tmp_dict[str(j)]=begin
    #         begin=begin+1
    #     values[i]=tmp_dict
    return values,mins

key_template_dict={}
# 对边进行编码
def handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0):
    data_processed_=[]
    obj={}
    temp_value=''
    tmplist=list(json_obj.keys())
    if 'event id' in tmplist:  # event
        temp_key='event id'
        eventid=re_values[temp_key][json_obj[temp_key]]
        temp_value='eventid:'+str(len(edges[2]))
        child=re_values['hash'][json_obj['childVertexHash']]
        parent=re_values['hash'][json_obj['parentVertexHash']]
        edges[2].append(child)  # 如果是边就存在 edges[2][3]中
        edges[3].append(parent)
    if 'hash' in tmplist:      # node_hash
        temp_key='hash'
        if 'pid' in tmplist:  # 如果是节点中的 Pid声明，就存在 edges[0][1]中
            edges[0].append(re_values[temp_key][json_obj[temp_key]]) # 该 json_obj 对应的 hash 对应的 index
            edges[1].append(re_values['pid'][json_obj['pid']])  # 该json_obj 对应的 pid 对应的 index
        temp_value='verteid:'+str(re_values[temp_key][json_obj[temp_key]]) # 该 json_obj 对应的 hash 对应的 index，并构建 verteid:index
        for temp_char in str(temp_value):   # 根据字符编码
            if temp_char not in char2id_dict:
                end=len(char2id_dict)+2
                char2id_dict[temp_char]=end
                id2char_dict[end]=temp_char
                data_processed_.append(end)
            else:
                data_processed_.append(char2id_dict[temp_char])
    data_processed_.append(0)  # 0作为分隔符？
    # 属性组合模板字典（不同json 可能有的属性组合不同，因为数据丢失）
    if ','.join(tmplist) not in key_template_dict.keys(): 
        key_template_dict[','.join(tmplist)]=len(key_template_dict.keys())
    key_indx=str(key_template_dict[','.join(tmplist)])
    if key_indx not in char2id_dict:
        end=len(char2id_dict)+2
        char2id_dict[key_indx]=end
        id2char_dict[end]=key_indx
        data_processed_.append(end)
    else:
        data_processed_.append(char2id_dict[key_indx])
    data_processed_.append(0)
    count=0
    for temp_key in tmplist:
        if temp_key=='hash':  # 上面处理过了
            continue  
        else:
            count=count+1
        if temp_key =='startTimestampNanos':
            temp_value=str(int(json_obj[temp_key])-mins[0])
        elif temp_key=='childVertexHash' or temp_key=='parentVertexHash':
            continue
            temp_value=str(re_values['hash'][json_obj[temp_key]])
        elif temp_key=='time':
            tt,order=translate_time(json_obj[temp_key])
            temp_value=str(tt)+','+str(order)
        else:
            temp_value=re_values[temp_key][json_obj[temp_key]]
        for temp_char in str(temp_value):
            if temp_char not in char2id_dict:
                end=len(char2id_dict)+2
                char2id_dict[temp_char]=end
                id2char_dict[end]=temp_char
                data_processed_.append(end)
            else:
                data_processed_.append(char2id_dict[temp_char])
        data_processed_.append(0)
    data_processed_.append(1)  # 1 作为结尾符
    return data_processed_,edges
import os
import pickle
edges=[]
edges.append([])
edges.append([])
edges.append([])
edges.append([])
error=0
re_values,mins=count()
print('finished')
data_processed=[]
reader=[]
data=[]
with open(args.input_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        data.append(i)
key=data[0]
data=data[1:]
for i in range(len(data)):
    json_obj={}
    tmpdata=data[i]
    for j in range(len(key)):
        if tmpdata[j]!='':
            json_obj[key[j]]=tmpdata[j]
    tmp_strr,edges=handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0)
    if tmp_strr=='':
        error=error+1
    else:
        data_processed.append(tmp_strr) 
reader=[]
data=[]
with open(args.input_path1, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        data.append(i)
key=data[0]
data=data[1:]
for i in range(len(data)):
    json_obj={}
    tmpdata=data[i]
    for j in range(len(key)-1):
        if tmpdata[j+1]!='':
            json_obj[key[j+1]]=tmpdata[j+1]
    tmp_strr,edges=handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0)
    if tmp_strr=='':
        error=error+1
    else:
        data_processed.append(tmp_strr) 


for i in re_values.keys():
    tmp_dict=re_values[i]
    tmp_list=list(range(len(tmp_dict.keys())))
    for j in tmp_dict.keys():
        tmp_list[tmp_dict[j]]=j
    re_values[i]=tmp_list
tmp_vec=re_values['event id']  # 记录 offset
for i in range(len(tmp_vec)):
    tmp_vec[i]=str(int(tmp_vec[i])-mins[1] )
re_values['event id']=tmp_vec
edges1=''   # 保存成字符串
for i in edges:
    edges1=edges1+str(i)+'\n'
f1=open(args.edge_file,'w')  # 保存成edges200m.npy
f1.write(edges1)
f1.close()
# exit()
np.save(args.edge_file,edges)
out = [c for item in data_processed for c in item]  # 把 data_processed 中的所有单个数字保存成一个列表
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
np.save(args.output_path, integer_encoded)  # 保存成vertex200m.npy
params = {'id2char_dict':id2char_dict,'char2id_dict':char2id_dict,'mins':mins,'re_values_dict':re_values,'key_template_dict':key_template_dict}
with open(args.param_file, 'w') as f:  # 保存超参（映射表）到 params.json 中
    json.dump(params, f, indent=4)

