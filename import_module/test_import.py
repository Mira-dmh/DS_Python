# test_import.py

############################################################# 
#	If we want to access the varialbe sand functions 
#	defined in some_module.py from another file in 
# 	the same directory, we could use the import statement
############################################################# 

##### Version 1: import a module
import some_module

pi = some_module.PI
print ('Version 1: PI = ', pi)

result = some_module.g(5, pi)
print ('Version 1: g(5, PI) = ', result, '\n')

##### Version 2: Create aliases by using the 'as' keyword 
import some_module as sm
from some_module import PI as pi, g as gfun

r1 = sm.f(pi)
print ('Version 3: pi = ', pi, 'sm.f(pi) = ', r1)

r2 = gfun(5, pi)
print ('Version 3: gfun(5, PI) = ', r2, '\n')

##### Version 3: import a function/constant/variable from a module
from some_module import f, g, PI

print ('Version 2: PI = ', PI)

result = g(5, PI)
print ('Version 2: g(5, PI) = ', result, '\n')





