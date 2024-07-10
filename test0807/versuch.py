import os
import subprocess
from odbAccess import *
from abaqusConstants import *
import numpy as np

odb = session.odbs['C:/Users/weiso/Desktop/DA/test0807/0.odb']
session.writeFieldReport(fileName='C:/Users/weiso/Desktop/DA/test0807/abaqus.csv', append=ON, 
    sortItem='Element Label', odb=odb, step=0, frame=100, 
    outputPosition=ELEMENT_NODAL, variable=(('E', INTEGRATION_POINT), ('S', 
    INTEGRATION_POINT), ), stepFrame=SPECIFY)


db = session.odbs['C:/Users/weiso/Desktop/DA/test0807/0.odb']
session.writeFieldReport(fileName='C:/Users/weiso/Desktop/DA/test0807/abaqus11.csv', append=ON, 
    sortItem='Element Label', odb=odb, step=0, frame=100, 
    outputPosition=INTEGRATION_POINT, variable=(('S', INTEGRATION_POINT, ((
    COMPONENT, 'S11'), (COMPONENT, 'S22'), (COMPONENT, 'S33'), (COMPONENT, 
    'S12'), (COMPONENT, 'S13'), (COMPONENT, 'S23'), )), ), stepFrame=SPECIFY)