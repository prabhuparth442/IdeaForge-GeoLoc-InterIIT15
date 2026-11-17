# stream the uncompressed tiled BigTIFF from stdin, block-average 4x, write a uint16 memmap (1.12 m per pixel)
import sys, struct, numpy as np
out=sys.argv[1]; f=sys.stdin.buffer; pos=0
def rd(n):
    global pos; b=f.read(n); pos+=len(b); return b
head=rd(1345365)
W,H,TS,F=72365,75856,256,4
typ=None; o=672780; nt=84051
offs=np.frombuffer(head[o:o+8*nt],dtype='<u8'); order=np.argsort(offs)
tw=(W+TS-1)//TS; th=(H+TS-1)//TS; assert tw*th==nt
m=np.lib.format.open_memmap(out,mode='w+',dtype=np.uint16,shape=(th*TS//F,tw*TS//F))
for k,i in enumerate(order):
    skip=int(offs[i])-pos
    if skip: rd(skip)
    a=np.frombuffer(rd(TS*TS*2),dtype='<u2').reshape(TS,TS).astype(np.float32)
    r,c=divmod(int(i),tw); q=TS//F
    m[r*q:(r+1)*q,c*q:(c+1)*q]=a.reshape(q,F,q,F).mean((1,3)).round().astype(np.uint16)
    if k%10000==0: print(k,nt,flush=True)
m.flush(); print('done',m.shape)
