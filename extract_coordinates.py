import numpy as np




#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
cg_start =-3
cg_end = 0

for start in range(0,19764,12):
    cg_start +=3
    cg_end += 3
    end = start+12
    com = data_bb[:,cg_start:cg_end]
    curent = data[:,start:end]
    CA = curent[:,3:6]
    N =  curent[:,:3]
    C = curent[:,6:9]
    O = curent[:,9:12]

    C = np.subtract(C,com)
    N = np.subtract(N,com)
    O = np.subtract(O,com)
    CA = np.subtract(CA,com)
    if start ==0:
        change = np.concatenate((N,CA,C,O),axis=1)

    else:
        temp = np.concatenate((N,CA,C,O),axis=1)
        change= np.concatenate((change,temp),axis=1)

    print(start, end, cg_start,cg_end)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
    
cg_start =-3
cg_end = 0

for start in range(0,19764,12):
    cg_start +=3
    cg_end += 3
    end = start+12
    com = bb[:,cg_start:cg_end]
    curent = predicted_aa[:,start:end]
    CA = curent[:,3:6]
    N =  curent[:,:3]
    C = curent[:,6:9]
    O = curent[:,9:12]

    C = np.add(C,com)
    N = np.add(N,com)
    O = np.add(O,com)
    CA = np.add(CA,com)
    if start ==0:
        change = np.concatenate((N,CA,C,O),axis=1)

    else:
        temp = np.concatenate((N,CA,C,O),axis=1)
        change= np.concatenate((change,temp),axis=1)

    print(start, end, cg_start,cg_end)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# norm factor
norm = 240
