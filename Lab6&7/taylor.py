from sympy import *
import numpy as np
import matplotlib.pyplot as plt

#### DEFINITION OF f(x,t) : (R^n,R) -> R^n #####
#List of strings // Coordonate : t, x_0 , x_1 , ...
f = ["-x_0"]
X_0 = [ Interval(1,1.2)]
###############################
X = ["x_{}".format(i) for i in range(len(f))] #Identité X


def parse_function(f : list) -> Matrix :
    """
        Convert a list of string f into a Matrix of type expression from sympy
    """
    return Matrix( [parse_expr(f_i) for f_i in f] )



def Lie_derivative(f : Matrix, g : Matrix) -> Matrix:
    """
        Compute the Lie derivative L_f(g)

        f,g must be two Matrixs of type expression containing variables t, x_0, x_1 ,...
    """

    n_f, n_g = len(f), len(g) #dimension of f and g
    x = list(symbols(" ".join(["t"] + ["x_{}".format(i) for i in range(n_f)]))) #x = [t, x_0, x_1, ...]  // List of symbols
    lie_matrix = [ g[i].diff(x[0]) for i in range(n_g)  ] #Matrix that will be returned. Initiliazed with the temporal partial derivative part

    for i in range(n_f):
        for j in range(n_g) : #Iterate over dimension of g
            #computing (dg_j/dx_i)*f_i
            lie_matrix[j] += g[j].diff(x[i+1])*f[i]  #x et f don't have same indice (i+1 and i) because the first element of x is t --> offset of 1

    return Matrix(lie_matrix)



def Lie_derivative_repeat(f : Matrix, g : Matrix, n : int) -> list :
    """
        Compute the Lie derivative repeatedly following the reccurent relation :
            - n = 1 ==>  L_f(g)
            - Else L_f( L_f(g)^{n-1}  )

        f,g must be two lists of type expression containing variables t, x_0, x_1 ,...
        n must be an interger

        Return a list of the n Lie_derivative
    """
    assert n >= 1 and "int" in str(type(n))

    n_lie_matrix = [Lie_derivative(f, g)]

    for i in range(1,n):
        n_lie_matrix.append(  Lie_derivative(f,n_lie_matrix[-1]) )

    return n_lie_matrix



def sum_it(a, b) -> Interval :
    """
        Takes two intervals [a, a'] and [b, b'] or scalar
        Returns their sum [a + b, a' + b']
    """
    if 'float' in str(type(a)) : #Case when a is a singleton
        a_inf, a_sup = a, a
    else :
        a_inf, a_sup = a.inf, a.sup
    if 'float' in str(type(b)) : #Case when b is a singleton
        b_inf, b_sup = b, b
    else :
        b_inf, b_sup = b.inf, b.sup
    return Interval( a_inf + b_inf  ,   a_sup  + b_sup     )


def prod_it(a, b) -> Interval :
    """
        Takes two intervals [a, a'] and [b, b'] or scalar
        Returns the product [ min xy , max xy]
    """
    if 'float' in str(type(a)) : #Case when a is a singleton
        a_inf, a_sup = a , a
    else :
        a_inf, a_sup = a.inf, a.sup
    if 'float' in str(type(b)) : #Case when b is a singleton
        b_inf, b_sup = b , b
    else :
        b_inf, b_sup = b.inf, b.sup

    return Interval(  min(a_inf*b_inf,a_inf*b_sup,a_sup*b_inf,a_sup*b_sup) ,   max(a_inf*b_inf,a_inf*b_sup,a_sup*b_inf,a_sup*b_sup)  )



def range_function_multivariant(f : Matrix , bounds_list : list) -> list :
    """
        Takes as input a function f (Matrix), a list of interval for each symbol
        Returns a list of interval --> range for each dimension of f

        This function is not very efficient. It has a complexity of n^2 K (where K is the complexity to compute a maximum/minimum)
        We assume here that f is differentiable beside X
    """
    list_interval = []
    n_f = len(f)
    if n_f == 1 : #if only one element, just a little exception because cannot do list(x_0) where x_0 is a symbol
        x = [symbols('x_0')]
    else :
        x = list(symbols(" ".join(["x_{}".format(i) for i in range(n_f)]))) #x = [x_0, x_1, ...]  // List of symbols

    for f_i in f:
        # To compute the minimum of f_i, we first minimize on the first component x_0, then on the second x_1, so on and so fourth...
        # Same for the maximum
        lower_bound, upper_bound = f_i,f_i
        for j in range(n_f) :
            lower_bound = minimum(lower_bound, x[j], bounds_list[j])
            upper_bound = maximum(upper_bound, x[j], bounds_list[j])
        list_interval.append( Interval(lower_bound, upper_bound) )

    return list_interval



def enclore_is_contained(B : list, B0 : list) -> bool :
    """
        B and B0 are two lists of interval

        Return True in all intervals B[i] of B are in B0[i] of B0
    """
    n_f = len(B)
    for i in range(n_f):
        if not B[i].is_subset(B0[i]) :
            return False
    return True



def priori_enclosures(I : list, h : float, a : float, f : Matrix) -> (float, Matrix):
    """
        Compute at the same time a priori bounding box and valid step size

        Takes as input I (a list of initial interval), h (float), a (float), f a function
        returns a valid over-approximation list interval + h
    """
    n_f = len(I) ## n_f has the same dimension as f and X

    #Initiliaze B
    B = []
    f_of_I = range_function_multivariant(f, I)
    for i in range(n_f) : #Iterate over each dimension
        try :
            B.append(  sum_it( I[i] ,  prod_it( Interval(0,h) , f_of_I[i] )  ) )
        except Exception as e :
            print("Error : {}".format(e))
            print("[INIT OF B] Step {} : range --> {}".format(i,f_of_I[i]))

    #Initiliaze new B := X_j + [0,h]f(B)
    new_B = []
    f_of_B = range_function_multivariant(f, B)
    for i in range(n_f) : #Iterate over each dimension
        try :
            new_B.append(  sum_it( I[i] ,  prod_it( Interval(0,h) , f_of_B[i] )  ) )
        except Exception as e :
            print("Error : {}".format(e))
            print("[INIT OF NEW_B] Step {} : range --> {}".format(i,f_of_B[i]))

    while ( not(enclore_is_contained(new_B, B)) ) :
        #Update B
        B = new_B.copy()
        for i in range(n_f) : #Iterate over each dimension
            try :
                B[i] =   sum_it( B[i] ,  prod_it( Interval(-a,a) , B[i] )  )
            except Exception as e :
                print("Error : {}".format(e))
                print("[UPDATE B] Step {} : range --> {}".format(i,B[i]))

        #Update of h
        h /= 2

        #Update new_B
        f_of_B = range_function_multivariant(f, B)
        for i in range(n_f) : #Iterate over each dimension
            try :
                new_B[i] = sum_it( I[i] ,  prod_it( Interval(0,h) , f_of_B[i] )  )
            except Exception as e :
                print("Error : {}".format(e))
                print("[UPDATE OF NEW_B] Step {} : range --> {}".format(i,f_of_B[i]))

    return h,B


def tightening(I : list, order : int, B : list, f : Matrix, X : Matrix) -> list :
    """
        Compute the second step : Tightening

        Takes as input I (a list of initial interval), order of the Taylor Model (int), B the enclure found at the previous step (list), the function
        returns a (order + 1)-long list of lists of n intervals in front of each Taylor devepment monome.
    """
    n_f = len(f)
    coef_list = [I] #Initialize list to return
    lie_derivative_list = Lie_derivative_repeat(f,X,order) #Generate the Lie derivative.


    #Calculate the range for each Lie derivative
    for i in range(order) :
        if i != order-1 : #Not the last iteration
            coef_list.append(  range_function_multivariant(lie_derivative_list[i], I)  )
        else : #Last iteration ==> Must be determined with B
            coef_list.append(  range_function_multivariant(lie_derivative_list[i], B)  )

    return coef_list


def reachable_state(coef_tightening_list : list, t : float) -> list :
    """
        Final step where we can evaluate with t from the approximation

        Takes as input coef_tightening_list the list of intervals returned by the function tightening (list), t the time where we want to evaluation (float)
    """
    t = float(t)
    n_f = len(coef_tightening_list[0])
    order = len(coef_tightening_list) - 1
    assert order >= 1 #Useless if order < 1

    #We use Horner's method
    #Initialize the part of the polynome evaluated
    current_interval = []
    for l in range(n_f): #Iterate over each dimension
        current_interval.append( sum_it(  prod_it(  (t/np.math.factorial(order)) , coef_tightening_list[-1][l] ) , prod_it( 1/np.math.factorial(order-1)  , coef_tightening_list[-2][l]) ) )

    #While loop
    i = order
    while (i != 1) :
        i -= 1
        for l in range(n_f): #Iterate over each dimension
            current_interval[l] = sum_it(  prod_it(  t , current_interval[l] ) , prod_it( 1/np.math.factorial(i-1),  coef_tightening_list[i-1][l]) )

    return current_interval



def make_step(f : Matrix, X : Matrix, I : list, order : int) -> (list, float) :
    """
        Compute a hole step in Taylor Model

        Returns the new interval for the next step (list) and the step h made
    """
    h,B = priori_enclosures( I = I , h = 1 , a = 0.1, f = f)
    tight =  tightening( I = I , order = order, B = B, f = f, X = X)
    return reachable_state(tight, h) , h


if __name__ == "__main__":
    f = parse_function(f)
    X = parse_function(X)

    T = 4
    t = 0

    X_up = []
    X_down = []
    Y = []

    m,n = X_0[0].sup, X_0[0].inf

    error = 0

    while t < T :
        X_0,h = make_step(f,X,X_0, 4)
        t += h
        X_up.append(X_0[0].sup)
        X_down.append(X_0[0].inf)
        Y.append(t)
        error += X_0[0].sup - X_0[0].inf + (m-n)*np.exp(-t)

        #print(t, X_0)

    print("error", error)
    Y = np.array(Y)
    X_exp = m*np.exp(-Y)
    X_exp2 = n*np.exp(-Y)
    plt.plot(Y , X_up, color = "r")
    plt.plot(Y , X_down, color = "b")
    plt.plot(Y , X_exp, color = "g")
    plt.plot(Y , X_exp2, color = "g")
    plt.show()
