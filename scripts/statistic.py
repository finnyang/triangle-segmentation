import math
import os

# directory of file
# rate the rate of samples been saved
def process(dir,file,save_file,rate):
    file_path=os.path.join(dir,file)
    fid=open(file_path)
    lines=fid.readlines()
    fid.close()
    path_ration_map={}
    for line in lines:
        line=line.strip().split(' ')
        path_ration_map[line[0]]=float(line[1])
    #for key in path_ration_map.keys():
    #    print key,path_ration_map[key]
    sort_map=sorted(path_ration_map.items(),key=lambda x:x[1],reverse=True)
    #print sort_map[5000]
    num=len(sort_map)
    reserve=math.floor(num*rate)
    print 'total %d images,reserver %d images'%(num,reserve)
    save_path=os.path.join(dir,save_file)
    fid=open(save_path,'w')
    for i in range(int(reserve)):
        fid.write(sort_map[i][0]+" "+str(sort_map[i][1])+'\n')
    fid.close()

if __name__=="__main__":
    process('/home/yang/triangle-segmentation/data','path_ratio.txt','reserve.txt',0.5)