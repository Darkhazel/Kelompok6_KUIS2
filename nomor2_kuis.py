from numpy import matrix, linalg

def f(x, y):
  return (2 - x)**2 + 200(y - x**2)**2


def f_dx(x, y):
  return


def f_dy(x, y):
  return 


def f_dx_dx(x, y):
  return 


def f_dx_dy(x, y):
  return  


def f_dy_dx(x, y):
  return 4*(x+y)


def f_dy_dy(x, y):
  return 12*y**2 + 4*x - 26

#Newton Method
class Newton:
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
      print(self.xyvector)
      self.determineDf()
      print(self.Df)
      self.determineH()
      print(self.H)
      print(self.inversH)
      self.determineNewVector()

      print(f"x{(i+1)} = ({self.x},{self.y})")
    print(f"f(x) = {f(self.x,self.y)}")




class SteepestDescent:
  def __init__(self, x, y, t, n):
    self.x = x
    self.y = y
    self.t = t
    self.n = n

  def determineT(self, vector, gradientMatrix):
    symbolizeX = vector[0,0] - gradientMatrix[0,0] * self.t
    symbolizeY = vector[1,0] - gradientMatrix[0,0] * self.t
    
    matGrad = matrix([[f_dx(symbolizeX,symbolizeY)], [f_dy(symbolizeX,symbolizeY)]])

    return matGrad

  def solve(self):
    for i in range(self.n):
      xVector = matrix([[self.x], [self.y]])
      print(xVector)
      gradientMatrix = matrix([[f_dx(self.x,self.y)], [f_dy(self.x, self.y)]])
      print(gradientMatrix)
      newXVector = xVector - self.determineT(xVector, gradientMatrix) * gradientMatrix
      x = newXVector[0,0]
      y = newXVector[1,0]
      print(f"x{i+1} = ({self.x}, {self.y})")



class PSO:
  def __init__(self, x, y, vX, vY, c, r, w, n):
    self.x = x
    self.y = y

    self.vX = vX
    self.vY = vY

    for i in range(1, len(x)):
      self.vX.append(vX[0])
      self.vY.append(vY[0])

    self.c = c
    self.r = r
    self.w = w
    self.n = n

    self.f = [0, 0, 0, 0]

    self.gBestX = None
    self.gBestY = None

    self.pBestX = [0, 0, 0, 0]
    self.pBestY = [0, 0, 0, 0]

    self.oldX = [0, 0, 0, 0]
    self.oldY = [0, 0, 0, 0]

    self.vectorX = [0, 0, 0, 0]
    self.v1 = [0, 0, 0, 0]
    self.vectorGBest = [0, 0, 0, 0]
    self.vectorPBest = [0, 0, 0, 0]

  def particle(self, x, y, vX, vY):
    for i in range(len(x)):
      self.x[i] = x[i]
      self.y[i] = y[i]
      self.vectorX[i] = matrix([[x[i]], [y[i]]])
    for i in range(len(vX)):
      self.vX[i] = vX[i]
      self.vY[i] = vY[i]
      self.v1[i] = matrix([[vX[i]], [vY[i]]])
    
    #Step 2
  def determineFxy(self):
    for i in range(len(self.x)):
      self.f[i] = f(self.x[i], self.y[i])

  def determineGBest(self):
    self.gBestX = self.gBestY = self.x[self.f.index(max(self.f))]
    self.vectorGBest = matrix([[self.gBestX], [self.gBestY]])

  def determinePBestIter1(self):
    for i in range(len(self.x)):
      self.pBestX[i] = self.x[i]
      self.pBestY[i] = self.y[i]
      self.vectorPBest = matrix([[self.pBestX[i], [self.pBestY[i]]]])

  def determinePBest(self):
    for i in range(len(self.x)):
      if f(self.oldX[i], self.oldY[i]) >= self.f[i]:
        self.pBestX[i] = self.oldX[i]
      else:
        self.pBestX[i] = self.x[i]

      if f(self.oldX[i], self.oldY[i]) >= self.f[i]:
        self.pBestY[i] = self.oldY[i]
      else:
        self.pBestY[i] = self.y[i]

  def updateV(self):
    for i in range(len(self.vX)):
      self.v1[i] = self.w * self.v1[i] + self.c[0] * self.r[0] * (self.v1[i] - self.vectorX[i]) + self.c[1] * self.r[1] * (self.vectorGBest - self.vectorX[i])
      self.vX[i] = self.v1[i][0,0]
      self.vY[i] = self.v1[i][1,0]

  def updateXY(self):
    for i in range(len(self.x)):
      self.oldX[i] = self.x[i]
      self.oldY[i] = self.y[i]
      self.vectorX[i] = self.vectorX[i] + self.v1[i]

      self.x[i] = self.vectorX[i][0,0]
      self.y[i] = self.vectorX[i][1,0]

  def solve(self):
    for i in range(self.n):
      print(f"{i+1}=======================================================")
      self.particle(self.x, self.y, self.vX, self.vY)
      self.determineFxy()
      self.determineGBest()
      self.determinePBestIter1() if i == 0 else self.determinePBest()
      self.updateV()
      self.updateXY()

      for i in range(len(self.x)):
        print(f"(x, y) = ({self.x[i]}, {self.y[i]})")

print("Newton : ")
newton = Newton(5,5,3)
newton.solve()

print("Steepest Descent : ")
steepest = SteepestDescent(5,5,1/4,3)
steepest.solve()

print("PSO : ")
pso = PSO([1, 3, -2, 5], [-1, 0, 2, 5], [0], [0], [1, 0.5], [1, 1], 1, 10)
pso.solve()