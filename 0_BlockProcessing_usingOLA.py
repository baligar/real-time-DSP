import numpy as np
from numpy.fft import rfft, irfft

HN = [1,1,1]
XN = [3,-1,0,1,3,2,0,1,2,1, 3,-1,0,1,3,2,0,7,8,9]

M = len(HN) # length of analysis/synth filter/ kernel
L = 4 # block size of incoming samples/ hop-size
N = L + M - 1

h_n = HN
x_n = XN
L_all = len(x_n)

print("L_all, L, M, N:",L_all, L, M, N)

# Step1: pad h_n with L-1 zeros
print(h_n)
h_n = h_n + [0]*(N-len(h_n))
print("padded h_n: ",h_n)

## Step 2: pad x1_n, x2_n.... chunks of size L, with M-1 zeros

# Step 2.1: Chunk x_n first into L blocks
chunks = []
for i in range(len(x_n)):
    if(i%L==0):
        #print(x_n[i:i+L])
        if(len(x_n[i:i+L]) != L):
            chunks.append(x_n[i:i+L] + [0]*(L - len(x_n[i:i+L])))
        else:
            chunks.append(x_n[i:i+L])
            
print(chunks)

# Step 2.2: Adding "M-1" padding above is possible, but making it a second sub-step here:
padded_chunks = []
for chunk_ in chunks:
    padded_chunks.append(chunk_ + [0]*(M-1))
print("padded x_ns: ",padded_chunks)

# Step3: Perfrom convolution op between xn and hn, to get yn
y_n = []
for chunk_ in padded_chunks:
    # You can either convolve in time domain
    #yi_n = np.convolve(chunk_, h_n)
    
    # or multiple in freq domain
    chunk_f = rfft(chunk_)
    h_n_f = rfft(h_n)
    yi_n_f = chunk_f * h_n_f
    yi_n = irfft(yi_n_f)
    
    y_n.append((yi_n[:N]).tolist())
    
print("y_n:",y_n)


# Step 4: Perform Overlap and Add. Just "add" the last M-1 from y1_n to first y2_n
print("-------Step4------")
new_y = []

a = y_n[0]
for i in range (len(y_n)-1):
    a = a + [0]*(N - (M-1)) # Zero pad later 
    b = [0]*(N - (M-1))*(i+1) + y_n[i+1] # Zero pad front
    print("OLA step",i+1)
    print(a, b)
    a = np.array(a) + np.array(b)
    print(a.tolist())
    a = a.tolist()

print("Number of OLA steps finished:", len(y_n)-1) 
print("\nfinal y(n):", np.int32(a))

print("\n\n-----Direct method------")
h_n = HN
x_n = XN

y_n = np.convolve(x_n, h_n)
print(y_n)
