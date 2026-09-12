import numpy as np, math
R=1800.0
S=[]
for x in np.arange(-R,R+1,20.0):
    for y in np.arange(-R,R+1,20.0):
        if x*x+y*y<=R*R: S.append((x,y))
S=np.array(S)
def cov(cs):
    d=np.full(len(S),1e18)
    for c in cs: d=np.minimum(d,np.hypot(S[:,0]-c[0],S[:,1]-c[1]))
    return d.max()
print("%-16s %8s %8s %8s" % ("布局","rho","最大未覆盖","路径长(m)"))
for k in (6,7,8):
    for rho in (1000,1100,1200,1300,1400,1500,1550,1600):
        cs=[(0.0,0.0)]+[(rho*math.cos(2*math.pi*i/k),rho*math.sin(2*math.pi*i/k)) for i in range(k)]
        v=cov(cs)
        # 路径：原点(中心点) -> 环上依次访问，环周长 k*2*rho*sin(pi/k)，加上中心到环 rho
        L=rho + k*2*rho*math.sin(math.pi/k)
        print("1中心+%-2d环      %8.0f %8.1f %8.0f   %s" % (k,rho,v,L,"OK" if v<=1000 else ""))
    print()
# 纯环布局
for k in (7,8,9):
    for rho in (1000,1100,1200,1300):
        cs=[(rho*math.cos(2*math.pi*i/k),rho*math.sin(2*math.pi*i/k)) for i in range(k)]
        v=cov(cs); L=rho + k*2*rho*math.sin(math.pi/k)
        print("纯%-2d环          %8.0f %8.1f %8.0f   %s" % (k,rho,v,L,"OK" if v<=1000 else ""))
    print()
