import operator
from collections import defaultdict
import pickle
import sys
import numpy as np
import json
import argparse
# 命令行参数解析
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
args = parser.parse_args()

import time
begin_time=time.time()
# 时间转换函数
def translate_time(time_):
    day=time_.split('.')
    order=int(day[1])
    day=int(day[0])
    return [day, order]
# 构建所有 key:values 对，属性:{值:索引} （除了部分值 timestampNanos 等）
def get_dict_allkeys_values(dict_a,values,mins):
        # if 'hash' not in dict_a.keys() and 'sequence' not in dict_a.keys():
        #     print(dict_a)
        #     exit()
        for x in range(len(dict_a)):
            temp_key = list(dict_a.keys())[x]
            temp_value = dict_a[temp_key]
            if temp_key=='timestampNanos':   # 处理 timestampNanos, 
                if int(dict_a[temp_key])<=mins[0]:
                    mins[0]=int(dict_a[temp_key])
            elif temp_key=='startTimestampNanos':
                if int(dict_a[temp_key])<=mins[1]:
                    mins[1]=int(dict_a[temp_key])
            elif temp_key=='childVertexHash' or temp_key=='parentVertexHash':
                continue
            else:
                if temp_key=='sequence':
                    if int(dict_a[temp_key])<=mins[2]:
                        mins[2]=int(dict_a[temp_key])
                if temp_key=='size':
                    if int(dict_a[temp_key])<=mins[3]:
                            mins[3]=int(dict_a[temp_key])
                if temp_key not in values.keys():  # temp_key不在 value.keys 中就重新创建一个
                    values[temp_key]={}
                if str(temp_value) not in values[temp_key].keys(): # temp_value 不在 values[temp_key].keys()中
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
# 日志数据处理，读取和处理输入文件，生成 values 和 mins 字典
def count():
    reader=[]
    data=[]
    recordd={}
    # 先处理 verterx 文件
    with open(args.input_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            data.append(i)
    key=data[0]
    data=data[1:]
    mins=[sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize] # 保存对应的最小值,[timestampNanos, startTimestampNanos, sequence, size]
    values={}
    data_tmp=[]
    counters_list=[]
    for i in range(len(data)):
        tmpdata=data[i]
        json_obj={}
        for j in range(2,len(key)):
            if tmpdata[j]!='':
                json_obj[key[j]]=tmpdata[j]
        values=get_dict_allkeys_values(json_obj,values,mins)

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
    print('end count')
    return values,mins

key_template_dict={}
# 处理每一条记录，将每条记录转成编码形式，并更新节点和边信息
def handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0):
    data_processed_=[]
    obj={}
    temp_value=''
    tmplist=list(json_obj.keys())
    strr_tmp=''
    if 'sequence' in tmplist:
        temp_key='sequence'
        # eventid=re_values[temp_key][json_obj[temp_key]]
        temp_value='eventid:'+str(len(edges[2]))
        child=re_values['hash'][json_obj['childVertexHash']]
        parent=re_values['hash'][json_obj['parentVertexHash']]
        edges[2].append(child)
        edges[3].append(parent)
    elif 'hash' in tmplist:
        temp_key='hash'
     #   if 'pid' in tmplist:
     #       edges[0].append(re_values[temp_key][json_obj[temp_key]])
     #       edges[1].append(re_values['pid'][json_obj['pid']])
        temp_value='verteid:'+str(re_values[temp_key][json_obj[temp_key]])
    else:
        temp_key='sequence'
        # eventid=re_values[temp_key][json_obj[temp_key]]
        temp_value='eventid:'+str(len(edges[2]))
        child=re_values['hash'][json_obj['childVertexHash']]
        parent=re_values['hash'][json_obj['parentVertexHash']]
        edges[2].append(child)
        edges[3].append(parent)
    strr_tmp=strr_tmp+str(temp_value)
    for temp_char in str(temp_value):
        if temp_char not in char2id_dict:
            end=len(char2id_dict)+2
            char2id_dict[temp_char]=end
            id2char_dict[end]=temp_char
            data_processed_.append(end)
        else:
            data_processed_.append(char2id_dict[temp_char])
    data_processed_.append(0)

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
        if temp_key=='hash':
            continue
        else:
            count=count+1
        if temp_key =='startTimestampNanos':
            temp_value=str(int(json_obj[temp_key])-mins[1])
        elif temp_key =='timestampNanos':
            temp_value=str(int(json_obj[temp_key])-mins[0])
        elif temp_key=='childVertexHash' or temp_key=='parentVertexHash':
            continue
            temp_value=str(re_values['hash'][json_obj[temp_key]])
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
        strr_tmp=strr_tmp+' '+str(temp_value)
    data_processed_.append(1)
    print(strr_tmp)
    return data_processed_,edges
import os
import pickle
edges=[]
edges.append([])
edges.append([])
edges.append([])
edges.append([])
error=0
# 主流程，处理所有数据并保留结果
re_values,mins=count()
print('finished')
data_processed=[]
reader=[]
data=[]
with open(args.input_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        data.append(i)
    print('test')
key=data[0]
data=data[1:]
for i in range(len(data)):
    json_obj={}
    tmpdata=data[i]
    for j in range(2,len(key)):
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
tmp_vec=re_values['sequence']
for i in range(len(tmp_vec)):
    tmp_vec[i]=str(int(tmp_vec[i])-mins[2] )
re_values['sequence']=tmp_vec

tmp_vec=re_values['size']
for i in range(len(tmp_vec)):
    tmp_vec[i]=str(int(tmp_vec[i])-mins[3] )
    re_values['size']=tmp_vec


edges1=''
for i in edges:
    edges1=edges1+str(i)+'\n'
f1=open(args.edge_file,'w')
f1.write(edges1)
f1.close()
# exit()
# 结果保存
np.save(args.edge_file,edges)
out = [c for item in data_processed for c in item]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
np.save(args.output_path, integer_encoded)
params = {'id2char_dict':id2char_dict,'char2id_dict':char2id_dict,'mins':mins,'re_values_dict':re_values,'key_template_dict':key_template_dict}
with open(args.param_file, 'w') as f:
    json.dump(params, f, indent=4)
end_time=time.time()
print(end_time-begin_time)
