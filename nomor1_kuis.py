import numpy as np

def f(x):
  return (-4 * x) * (np.sin(x))

#f'(x)
def f_(x):
  return -4 * np.sin(𝑥) - 4*𝑥 * np.cos(𝑥)
#f''(x)
def f__(x):
  return (-8*np.cos(𝑥)) + (4*𝑥 * np.sin(𝑥))

def gap():
  print("==========================")

#Newton Method
class Newton:
  def __init__(self, x, n):
    self.x = x
    self.n = n

  def solve(self):
    print(f"x0 = {self.x}")
    print(f"f(x) = {f(self.x)}")
    for i in range(0, self.n):
      gap()  
      #rebuild nilai x menggunakan rumus xi - f'(x)/f''(x)
      self.x = self.x - (f_(self.x)/f__(self.x)) 
      print(f"x{i+1} = {self.x}")
    print(f"f(x) = {f(self.x)}")



#steepestDescent
class SteepestDescent:
  def __init__(self, x, t, n):
    self.x = x
    self.t = t
    self.n = n
    
  def solve(self):
    print(f"x0 = {self.x}")
    print(f"f(x) = {f(self.x)}")
    for i in range(0, self.n):
      gap()
      # Memperbaharui nilai x dengan menggunakan rumus x(i+1) = xi + (t * f'(x))
      self.x = self.x + (self.t * f_(self.x))
      print(f"f'(x{i+1}) = {f_(self.x)}")
      print(f"x{i+1} = {self.x}")
    print(f"f(x) = {f(self.x)}")



class PSO:
  #Step 1
  def __init__(self, x, v, c, r, w, n):
    self.x = x
    self.v0 = v
    self.c = c
    self.r = r
    self.w = w
    self.n = n

    self.gBest = 0
    self.pBest = []
    self.fxi = []
    self.v1 = [0,0,0]
    self.oldX = [0,0,0]
    
  #Step 2 menentukan F(xi)
  def detFxi(self):
    self.fxi = [f(x) for x in self.x]
    
  #step 3 Menentukan Gbest
  def detGBest(self):
    self.gBest = self.x[self.fxi.index(max(self.fxi))]

  #Step 4 Menentukan PBest 
  def detPBest(self):
    if self.pBest == []: #untuk iterasi 1
      self.pBest = [x for x in self.x]
    else: #untuk iterasi selanjutnya
      for i in range(len(self.x)):
        if f(self.x[i]) > f(self.oldX[i]):
          self.pBest[i] = self.x[i]
        else:
          self.pBest[i] = self.oldX[i]
  
  #Step 5 Memperbaharui nilai v
  def updateV(self):
    for i in range(len(self.v1)):
      self.v1[i] = (self.w * self.v1[i]) + (self.c[0]*self.r[0]*(self.pBest[i] - self.x[i])) + (self.c[1]*self.r[1]*(self.gBest - self.x[i]))

  #Step 6 Memperbaharui nilai x
  def updateX(self):
    for j in range(len(self.oldX)):
      self.oldX[j] = self.x[j]
    for i in range(len(self.x)):
      self.x[i] = self.x[i] + self.v1[i]

  def solve(self):
    print(f"x : {self.x}")
    print(f"f(x) = {f(self.x)}")
    for i in range(self.n):
      print(f"iter : {i+1}=======================================================")
      self.detFxi()
      self.detGBest()
      self.detPBest()
      self.updateV()
      self.updateX()
      print(f"x : {self.x}")
    print(f"f(x) = {f(self.x)}")

#main
print("Newton : ")
newton = Newton(np.pi/2, 3)
newton.solve()

print("\nSteepest Descent : ")
steepest = SteepestDescent(np.pi/2, 1/4, 3)
steepest.solve()

print("\nPSO : ")
pso = PSO(np.array([round(0), np.pi/2, np.pi]), 0, [1/2, 1], [1/2,1/2], 1, 3)
pso.solve()
