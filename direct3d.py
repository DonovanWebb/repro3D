import math
import numpy as np

def r(x):
    return np.round(x,3)

# Only measurable data:

S1v2 = (77-4) * 180/150
S2v3 = 0 * 180/150
S3v1 = (91-26) * 180/150

A = (S1v2-S2v3) * 2*math.pi/360  # angle between common tilt axes C12 and C31
B = (S2v3-S3v1) * 2*math.pi/360  # angle between common tilt axes C12 and C23
C = (S3v1-S1v2) * 2*math.pi/360  # angle between common tilt axes C23 and C31

A = (-71 /300) * 2*math.pi  # angle between common tilt axes C12 and C31
B = (46 /300) * 2*math.pi  # angle between common tilt axes C12 and C31
C = (198 /300) * 2*math.pi  # angle between common tilt axes C12 and C31
#######################################################

# Finding Euler angles Be and Al between projections 1,2,3

Be31 = A
Be23 = B
sin_Al23 = r((math.cos(C) - math.cos(Be23)*math.cos(Be31)) / (math.sin(Be23)*math.sin(Be31)))
Al23 = r(math.asin(sin_Al23))

print(r((Be31,Be23,Al23)))

C12 = (0,0,1) # Assume
C31 = (0, math.sin(Be31), math.cos(Be31)) # Assumme in yz plane
C31 = C31 / np.sqrt(C31[0]**2+C31[1]**2+C31[2]**2)
C23 = (math.sin(Be23)*math.cos(Al23), math.sin(Be23)*math.sin(Al23), math.cos(Be23))
C23 = C23 / np.sqrt(C23[0]**2+C23[1]**2+C23[2]**2)

D1 = np.cross(C31,C12)
D1 = D1 / np.sqrt(D1[0]**2+D1[1]**2+D1[2]**2)
D2 = np.cross(C12,C23)
D2 = D2 / np.sqrt(D2[0]**2+D2[1]**2+D2[2]**2)
D3 = np.cross(C23,C31)
D3 = D3 / np.sqrt(D3[0]**2+D3[1]**2+D3[2]**2)

# D1
D1Be = - math.acos(D1[2])
if math.sin(D1Be) == 0:
    D1Al = 0
else:
    D1Al = 0.5*math.asin(2*D1[0]*D1[1]/(math.sin(D1Be)**2))

D1check = r((math.sin(D1Be)*math.cos(D1Al), math.sin(D1Be)*math.sin(D1Al), math.cos(D1Be)))

# D2
D2Be = - math.acos(D2[2])
if math.sin(D2Be) == 0:
    D2Al = 0
else:
    D2Al = 0.5*math.asin(2*D2[0]*D2[1]/(math.sin(D2Be)**2))

D2check = r((math.sin(D2Be)*math.cos(D2Al), math.sin(D2Be)*math.sin(D2Al), math.cos(D2Be)))

# D3
D3Be = math.acos(D3[2])
if math.sin(D3Be) == 0:
    D3Al = 0
else:
    D3Al = 0.5*math.asin(2*D3[0]*D3[1]/(math.sin(D3Be)**2))

D3check = r((math.sin(D3Be)*math.cos(D3Al), math.sin(D3Be)*math.sin(D3Al), math.cos(D3Be)))

def rad_deg(x):
    return x*360/(2*math.pi)

print('C12: ', C12)
print('C31: ', C31)
print('C23: ', C23)
print('D1: ', r(D1))
print('D1 check: ', D1check)
print('D2: ', r(D2))
print('D2 check: ', D2check)
print('D3: ', r(D3))
print('D3 check: ', D3check)

print("(DXAl, DXBe)")
print(r((rad_deg(D1Al),rad_deg(D1Be))))
print(r((rad_deg(D2Al),rad_deg(D2Be))))
print(r((rad_deg(D3Al),rad_deg(D3Be))))
