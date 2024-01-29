import torch


def ndarray_tostring(array):
    string = ""
    for item in array:
        string += str(item.item()).strip()+" "
    return string+"\n"

def save_to_file(node_list_u,node_list_v,model_path,userlistIndex,itemlistIndex):
    with open(model_path[0],"w") as fw_u:
        idx=0
        for u in node_list_u:
            fw_u.write('u'+str(userlistIndex[idx].item())+" "+ ndarray_tostring(u))
            idx+=1
    with open(model_path[1],"w") as fw_v:
        idx=0
        for v in node_list_v:
            fw_v.write('i'+str(itemlistIndex[idx].item())+" "+ndarray_tostring(v))
            idx+=1

def read_file(uEmd,vEmd):
    u = {}
    udata = []
    with open(uEmd,"r") as fw_u:
        line = fw_u.readline()  
        while line:
            filedata = line.strip().split(" ")
            uemd = filedata[1:]
            idx = int(float(filedata[0][1:]))
            uVectors = list([ float(i) for i in uemd ])
            if u.get(idx) == None:
                u[idx] = []
            u[idx]=uVectors
            udata.append(uVectors)
            line = fw_u.readline()  
    vdata = []
    with open(vEmd,"r") as fw_v:
        line = fw_v.readline()
        while line:
            filedata = line.strip().split(" ")
            vemd = filedata[1:]
            idx = int(float(filedata[0][1:]))
            vVectors = list([float(i) for i in vemd])
            vdata.append(vVectors)
            line = fw_v.readline()
    return udata,vdata

