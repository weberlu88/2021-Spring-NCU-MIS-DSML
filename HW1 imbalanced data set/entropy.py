def balance(seq):
    from collections import Counter
    from numpy import log
    
    n = len(seq)
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    k = len(classes)
    # print('classes:', k)
    
    H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)

seq = [1, 2, 3, 4]
seq = [1, 1, 1, 1, 6, 6, 6, 6, 6, 6] # 4:6
seq = [1, 1, 1, 6, 6, 6, 6, 6, 6, 6] # 3:7
# seq = [1, 6, 6, 6, 6, 6, 6, 6, 6, 6] # 1:9
entropy = balance(seq)
print(entropy)