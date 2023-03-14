import numpy as np

color_map = {
0 : (64, 128, 64),	
1 : (192, 0, 128),	
2 : (0, 128, 192),
3 : (0, 128, 64),	
4 : (128, 0, 0),		
5 : (64, 0 ,128),	
6 : (64, 0, 192), 
7 : (192, 128, 64),	
8 :(192, 192, 128),	
9 :(64 ,64 ,128),	
10 : (128, 0, 192),	
11 :(192, 0, 64),	
12 :(128, 128, 64),	
13 :(192, 0, 192),	
14 : (128, 64, 64),	
15 : (64, 192, 128),	
16 :(64, 64, 0),	
17 : (128, 64, 128),	
18 :(128, 128, 192),	
19 : (0, 0, 192),		
20 : (192, 128, 128),	
21 : (128, 128, 128),	
22 : (64, 128, 192),	
23 : (0, 0, 64),		
24 : (0, 64, 64),		
25 : (192, 64, 128),	
26 : (128, 128, 0),	
27 : (192, 128, 192),	
28 : (64, 0, 64),		
29 : (192, 192, 0),	
30 : (64, 192, 0),	
50 : (0, 0, 0),		

}

def decode_label(encode_lable):
    r = np.zeros((encode_lable.shape[0], encode_lable.shape[1],1),dtype=np.uint8)
    g = np.zeros((encode_lable.shape[0], encode_lable.shape[1],1),dtype=np.uint8)
    b = np.zeros((encode_lable.shape[0], encode_lable.shape[1],1),dtype=np.uint8)
    for k,v in color_map.items(): # k = 4 v = (128,0,0)
        r[encode_lable == k] = v[0]
        g[encode_lable == k] = v[1]
        b[encode_lable == k] = v[2]
    rgb = np.concatenate((r,g,b), axis=2)
    return rgb