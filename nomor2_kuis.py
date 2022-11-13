from numpy import matrix, linalg
import numpy as np
import sympy as sp

def f(x, y):
  return (x**2+y-11)**2 + (x+y**2-7)**2


def f_dx(x, y):
  return 4*x*(x**2+y-11) + 2*(x+y**2-7)


def f_dy(x, y):
  return 2*(y+x**2-11) + 4*y*(y**2+x-7)


def f_dx_dx(x, y):
  return 12*x**2 + 4*y - 42


def f_dx_dy(x, y):
  return 4*(x+y)


def f_dy_dx(x, y):
  return 4*(x+y)


def f_dy_dy(x, y):
  return 12*y**2 + 4*x - 26


class NewtonMethod:
  def __init__(self, x, y, n):
    self.x = x
    self.y = y
    self.n = n

    self.xyvector = None
    self.Df = None
    self.H = None
    self.inversH = None

  def determineXyVector(self):
    self.xyvector = matrix([[self.x], [self.y]])

  def determineDf(self):
    self.Df = matrix([[f_dx(self.x, self.y)], [f_dy(self.x, self.y)]])

  def determineH(self):
    self.H = matrix([[f_dx_dx(self.x, self.y), f_dx_dy(self.x, self.y)],
                      [f_dy_dx(self.x, self.y), f_dy_dy(self.x, self.y)]])
    self.inversH = linalg.inv(self.H)

  def determineNewVector(self):
    vctN = self.xyvector - self.inversH * self.Df

    self.x = vctN[0, 0]
    self.y = vctN[1, 0]

  def solve(self):
    print(f"x0 = ({self.x},{self.y})")
    print(f"f(x) = {f(self.x, self.y)}")
    for i in range(self.n):
      print("==========================================")
      self.determineXyVector()
      self.determineDf()
      self.determineH()
      self.determineNewVector()

      print(f"x{(i+1)} = ({self.x},{self.y})")
    print(f"f(x) = {f(self.x,self.y)}")

class SteepestDescent:
  def __init__(self, xn, n, mi):
    self.xn = xn
    self.n  = n
    self.mi = mi
    self.t  = None

    self.xyVector = None
    self.Df = None
  def determineXYVector(self):
    self.xyVector = matrix([[self.x], [self.y]])

  def determineDf(self):
    self.Df = matrix([[f_dx(self.x, self.y)], [f_dy(self.x, self.y)]])
  
  def determineT(self):
    t = sp.Symbol("t")
    newXn = self.xn - t * self.Df
    sf_newXn_matrix = np.matrix([[f_dx(newXn[0, 0],newXn[1, 0])], [f_dy(newXn[0, 0], newXn[1, 0])]])
    equation = sp.Eq(-(sf_newXn_matrix[0, 0] * self.Df[0, 0]+ sf_newXn_matrix[1, 0] * self.Df[1, 0]), 0)
    return float(list(sp.solveset(equation))[0])


  def updateXY(self):
    newXYVector = self.xyVector - self.t * self.Df
    self.x = newXYVector[0,0]
    self.y = newXYVector[1,0]

  def solve(self):
    if self.mi == 0:
        self.mi = self.n
        print("Iteration 0")
    print("F(X0, Y0) =", f(self.xn[0], self.xn[1]))
    print("(X,Y)0 =", self.xn)
    try :
      for i in range(self.n):
        x = self.xn[0] 
        y = self.xn[1]
        self.xyVector = matrix([[self.x], [self.y]])
        self.Df = matrix([[f_dx(self.x, self.y)], [f_dy(self.x, self.y)]])
        if t != None :
            t = 0.25
        else : t = self.determineT(self.xyVector,self.Df)
        print(t)
        new_xn = self.xyVector-round(t, 2)*self.Df
        print()
        print("Iteration "+str(self.mi-self.n+1))
        print("F(X"+str(self.mi-self.n+1)+",Y"+str(self.mi-self.n+1)+") =", f(new_xn[0,0],new_xn[1, 0]))
        print("(X,Y)"+str(self.mi-self.n+1), "=",new_xn.flatten())
        if self.n == 1 : 
            x = new_xn[0,0] 
            y = new_xn[1,0]
            print()
            print("nilai minimum = ", f(x, y))
        else :
            self.xn = [new_xn[0, 0], new_xn[1, 0]]
        SteepestDescent(new_xn, self.n-1, self.mi)
    except :
        print()
        print("Tidak terdefinisi pada X" +
        str(self.mi-self.n+1))
        return False


class PSO:
  def __init__(self, x, y, v, c, r, w):
    self.x = x
    self.y = y
    self.v = v
    self.c = c
    self.r = r
    self.w = w

    self.gBest = 0
    self.gBestY = 0
    self.pBest = []
    self.pBestY = []
    self.fxi = []
    self.v1 = [0,0,0]
    self.v1y = [0,0,0]
    self.oldX = [0,0,0]
    self.oldY = [0,0,0]
    
  #Step 2 menentukan F(xi)
  def determineFxi(self):
    self.fxi = [f(self.x[i], self.y[i]) for i in range(len(self.x))]
    # print(f"fxi : \n{self.fxi}")
    
  #step 3 Menentukan Gbest
  def determineGBest(self):
    self.gBest = self.x[self.fxi.index(min(self.fxi))]
    self.gBestY = self.y[self.fxi.index(min(self.fxi))]
    # print(f"gBestX : {self.gBest}")
    # print(f"gBestY : {self.gBestY}")

  #Step 4 Menentukan PBest 
  def determinePBest(self):
    if self.pBest == []: #untuk iterasi 1
      self.pBest = [x for x in self.x]
      self.pBestY = [y for y in self.y]
    else: #untuk iterasi selanjutnya
      for i in range(len(self.x)):
        if f(self.x[i], self.y[i]) <= f(self.oldX[i], self.oldY[i]):
          self.pBest[i] = self.x[i]
          self.pBestY[i] = self.y[i]
        else:
          self.pBest[i] = self.oldX[i]
          self.pBestY[i] = self.oldY[i]
    # print(f"pBestX : \n{self.pBest}")
    # print(f"pBestY : \n{self.pBestY}")
       
  
  #Step 5 Memperbaharui nilai v
  def updateV(self):
    for i in range(len(self.v1)):
      self.v1[i] = (self.w * self.v1[i]) + (self.c[0]*self.r[0]*(self.pBest[i] - self.x[i])) + (self.c[1]*self.r[1]*(self.gBest - self.x[i]))
      self.v1y[i] = (self.w * self.v1y[i]) + (self.c[0]*self.r[0]*(self.pBest[i] - self.y[i])) + (self.c[1]*self.r[1]*(self.gBest - self.y[i]))
    # print(f"v1x : \n{self.v1}")
    # print(f"v1y : \n{self.v1y}")

  #Step 6 Memperbaharui nilai x
  def updateX(self):
    for j in range(len(self.oldX)):
      self.oldX[j] = self.x[j]
      self.oldY[j] = self.y[j]
    for i in range(len(self.x)):
      self.x[i] = self.x[i] + self.v1[i]
      self.y[i] = self.y[i] + self.v1y[i]

  def showXandFx(self, i):
    print(f"x{i+1} : ", end="")
    for p in range(len(self.x)):
      print(f"{self.x[p], self.y[p]}", end="")
      if p == len(self.x) - 1:
        print()
      
    print(f"f(x) : ", end="")  
    for q in range(len(self.x)):
      print(f"({f(self.x[q], self.y[q])})", end="")
      if q == len(self.x) - 1:
        print()
    

  def solve(self, n):
    print(f"x0 : ", end="")
    for j in range(len(self.x)):
      print(f"{self.x[j], self.y[j]}", end="")
      if j == len(self.x) - 1:
        print()
    for i in range(n):
      print(f"=======================================================")
      self.determineFxi()
      self.determineGBest()
      self.determinePBest()
      self.updateV()
      self.updateX()

      self.showXandFx(i)

print("Newton : ")
newton = NewtonMethod(1,1,3)
newton.solve()

print("Steepest Descent : ")
steepest = SteepestDescent([1, 1], 3, 0)
steepest.solve()
print("PSO : ")
pso = PSO([1, -1, 2], [1, -1, 1], [0,0], [1, 0.5], [1, 1], 1)
pso.solve(3)
