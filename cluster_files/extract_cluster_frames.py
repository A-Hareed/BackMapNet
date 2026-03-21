import pickle
import itertools

def fix_lst(lst):
    result = []
    for i in reversed(lst):
        if i =='|':
            result = [int(i) for i in result]
            return result
        else:
            result.append(i)


log_file = 'cluster.log'
counter =0
lst = []
clusters= {}
keys_clust = []
with open(log_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split()
        if len(line) < 1:
            pass
        elif line[0] == 'cl.':
            counter = 1
  #          print(line)
        elif counter==1:
            if line[0].isdigit() == True:
 #               print(line[0])
                keys_clust.append(line[0])
                if len(lst)>0:
                    clusters[lst[0][0]] = lst
                lst = []
                lst.append(line)
            else:
                tmp_lst = [int(i) for i in line[3:]]
                lst.append(tmp_lst)
        if line == lines[-1].split():
            if keys_clust[-1] != list(clusters.keys())[-1]:
                clusters[keys_clust[-1]] = lst
#                print(clusters[keys_clust[-1]])
            else:

                clusters[keys_clust[-1]] = lst
#
#with open('clusters_2J4A_Thyroid_hormone_receptor.pkl', 'wb') as file:
#    pickle.dump(clusters, file)



fixed_dict = {}
for key, value in clusters.items():
    updated_lst = fix_lst(clusters[key][0])
    flattened_list = list(itertools.chain(*clusters[key][1:]))
    updated_lst.extend(flattened_list)
    fixed_dict[key] = updated_lst



for cluster_id, frames in fixed_dict.items():
    print(f'Cluster {cluster_id}: {frames}')
