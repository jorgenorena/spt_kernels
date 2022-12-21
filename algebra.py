#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 08:25:33 2021

@author: jorgenorena
"""


class expr ():
 
    def __init__(self,*args): 
        self.args = args
        self._mhash = None
 
    def __hash__(self):
        h = self._mhash 
        if h is None:
            h = hash((type (self).__name__,) + \
                     tuple (self._hashable_content()))
            self._mhash = h
        return h 
 
    def _hashable_content(self):
        return self.args
 
    # Maybe move away from this class?
    def surf(self, args): 
        """
        Recorre los argumentos de una tupla, se usa en operaciones 
        asociativas. 
        """
        list_args=[]
        for argument in args:
            if isinstance(argument, type(self)): 
                for sub_argument in argument.args:
                    list_args.append(sub_argument)
            else: 
                list_args.append(argument)
        return list_args
     
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return mult(const(other), self)
        return mult(self,other)

    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return plus(const(other), self)
        return plus(self,other)

    def __radd__(self, other):
        return self.__add__(other)
 
    def __sub__(self, other):
        return plus(self, const(-1)*other)
 
    def __rsub__(self, other):
        return plus(const(-1)*self, other)
 
    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return fraction(self, const(other))
        return fraction(self, other)

    def __rtruediv__(self,other):
        if isinstance(other, int) or isinstance(other, float):
            return fraction(const(other), self)
        return fraction(other, self)
 
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.args == other.args
        else:
            return False
 
    def __ne__(self, other):
        return not(self == other)
 
    def srepr(self):
        return type(self).__name__ \
            + '(' + ','.join(repr(a) for a in self.args) + ')'
            
    def __repr__(self):
        return self.repr()
    
    def __pow__(self, other):
        return power(self, other)
    
    def __rpow__(self, other):
        return power(other, self)
 
    def latex(self):
        return self.__str__()
    
    def _separate_const(self):
        return const(1), self
    
    
class var (expr):
    
    def __init__(self,*args): 
        self.args= args
        self.name =args[0] 
        self._mhash = None
         
    def __str__(self):
        return self.name
    
    def val (self):
        return self.name
    
    def deriv(self, x):
        if x.name == self.name:
            return const(1)
        else:
            return const(0)
    
    
class const (expr): 
    
    def __init__(self,*args):
        self.args = args
        self._mhash = None

    def __str__(self):
        return str (self.args[0])  

    def val (self):
        return self.args[0]

    def is_integer(self):
        if isinstance(self.val(), int):
            return True
        elif isinstance(self.val(), float):
            return self.val().is_integer()
        else:
            return False
        
    def _separate_const(self):
        return self, const(1)
    
    def deriv(self, x):
        return const(0)
        

class const_fraction (const):
    
    def __new__(cls, numer, denom):
        
        if denom == const(0) or denom == 0:
            raise ZeroDivisionError('Division by zero')
        
        instance = super(const_fraction, cls).__new__(cls)
        
        numv = numer.val() if isinstance(numer, const) else numer
        denv = denom.val() if isinstance(denom, const) else denom
        
        if isinstance(numv, int) and isinstance(denv, int):
            gcd = _gcd(numv, denv)
            instance.numer = int(numv/gcd)
            instance.denom = int(denv/gcd)
        else:
            instance.numer = numer
            instance.denom = denom
            
        if instance.denom == const(1) or instance.denom == 1:
            return instance.numer
        else:
            return instance
        
    def __init__(self, numer, denom):
        self.args =  (numer, denom)
        self._mhash = None
        
    def __str__(self):
        return '(' + str(self.numer) + '/' + str(self.denom) + ')'
    
    def val(self):
        return self.numer/self.denom


def _const_sum(*args):
    
    a_fraction = False
    a_simple = True
    for a in args:
        if isinstance(a, const_fraction):
            a_fraction = True
        elif not(a.is_integer()):
            a_simple = False
            break

    if a_fraction and a_simple:
        result = _sum_fractions(*args)
    else:
        result = sum(a.val() for a in args)
    return result

def _sum_fractions(*args):

    numers = [(a.numer if isinstance(a, const_fraction) else a.val())
              for a in args]
    denoms = [(a.denom if isinstance(a, const_fraction) else 1)
              for a in args]
    denom = _prod(denoms)
    numer = sum([int((numers[i]*denom)/denoms[i]) for i in range(len(args))])

    return const_fraction(numer, denom)


class plus (expr): 

    def __new__(cls, *args):
        instance = super(plus, cls).__new__(cls)
        instance.args = instance._order_p(args)
        if len(instance.args) == 1:
            return instance.args[0]
        else:
            return instance

    def __init__(self, *args):
        self._mhash = None

    def _order_p(self,args):
        consts = []
        not_consts = []
        for element in super().surf(args):
            if isinstance(element,const):
                consts.append(element)
            else: 
                not_consts.append(element)
        not_consts.sort(key=hash) 
        total = _const_sum(*consts)
        if total != 0:
            not_consts.insert(0,const(total))
        result = self._simplify(not_consts)
        return result

    def _simplify(self, args):
        previous = None
        c = 0
        terms = []
        def key(term):
            ci, t = _separate_const(term)
            return hash(t)
        args.sort(key=key)
        for term in args:
            ci, current = _separate_const(term)
            if current != previous:
                if previous != None:
                    terms.append(c*previous)
                c = ci
                previous = current
            else:
                c += ci
        terms.append(c*current)
        return tuple(terms)

    def __str__(self):
        return  '(' + ' + '.join(str(arg) for arg in self.args) + ')' 

    def __hash__(self):
        if self._mhash == None:  
            self._mhash= super().__hash__()
        return self._mhash

    def latex(self):
        return '(' + ' + '.join(arg.latex() for arg in self.args) + ')'

    def expanded(self):
        new_args = tuple(expand(i) for i in self.args)
        return plus(*new_args)
    
    def deriv(self, x):
        return sum(map(lambda t: derivative(t, x), self.args))
    
    
def _separate_const(term):
    try: 
        return term._separate_const()
    except AttributeError:
        if isinstance(term, int) or isinstance(term, float):
            return const(term), const(1)
        else:
            raise AttributeError("%s object has no attribute"\
                                 " '_separate_const'"%(type(term).__name__))
    

def _const_mult(*args):
    
    a_fraction = False
    a_simple = True
    for a in args:
        if isinstance(a, const_fraction):
            a_fraction = True
        elif not(a.is_integer()):
            a_simple = False
            break

    if a_fraction and a_simple:
        result = _mult_fractions(*args)
    else:
        result = _prod(a.val() for a in args)
    return result

def _mult_fractions(*args):

    numers = [(a.numer if isinstance(a, const_fraction) else a.val())
              for a in args]
    denoms = [(a.denom if isinstance(a, const_fraction) else 1)
              for a in args]
    denom = _prod(denoms)
    numer = _prod(numers)

    return const_fraction(numer, denom)
   

def _separate_exp(term):
    if isinstance(term, power):
        return term.exponent, term.base
    else:
        return const(1), term
    
class mult (expr):  

    def __new__(cls, *args):
        instance = super(mult, cls).__new__(cls)
        instance.args = instance._order_m(args)
        if len(instance.args) == 1:
            return instance.args[0]
        else:
            return instance
  
    def __init__(self,*args):
        self._mhash = None
    
    def _order_m(self,args):  
        consts=[]
        not_consts=[]
        for element in super().surf(args):
            if isinstance(element,const):
                consts.append(element)
            elif isinstance(element, int) or isinstance(element, float):
                consts.append(const(element))
            else: 
                not_consts.append(element)
        if len(consts) == 0:
            c = 1
        else:
            c = _const_mult(*consts)
            
        if c == 0:
            return (const(0),)
        elif c != 1:
            not_consts.insert(0,const(c))
        if len(not_consts) == 0:
            return (const(1),)
        else:
            return  self._simplify(not_consts)
        
    def _simplify(self, args):
        previous = None
        e = 0
        terms = []
        def key(term):
            ei, t = _separate_exp(term)
            return hash(t)
        args.sort(key=key)
        for term in args:
            ei, current = _separate_exp(term)
            if current != previous:
                if previous != None:
                    terms.append(previous**e)
                e = ei
                previous = current
            else:
                e += ei
        terms.append(current**e)
        return tuple(terms)

        
    def __eq__(self,other): 
        if isinstance(other, mult):
            return hash(self) == hash(other)
        else:
            return False


    def expanded(self):

        exp_args = tuple(expand(i) for i in self.args)

        plus_var = []
        not_plus_var = []
        #new_args = []

        for element in exp_args:
            if isinstance(element, plus):
                plus_var.append(element.args)
            else: 
                not_plus_var.append(element)

        if len(plus_var)==0:
            return mult(*tuple(not_plus_var))
        else:
            right=mult(*tuple(not_plus_var)) 
            left=plus(*tuple(_distribute_list(plus_var)))
            new_elements=[mult(right, left_element) for 
                          left_element in left.args]
        
        return plus(*tuple(new_elements)) 
  
    def __str__(self):
        return '*'.join(str(arg) for arg in self.args)

    def __hash__(self):
        if self._mhash == None:  
            self._mhash= super().__hash__()
        return self._mhash

    def latex(self):
        return ' '.join(arg.latex() for arg in self.args)
    
    def _separate_const(self):
        if isinstance(self.args[0], const):
            return self.args[0], mult(*self.args[1:])
        else:
            return const(1), self
        
    def deriv(self, x):
        result = 0
        args = list(self.args)
        for i, t in enumerate(self.args):
            result += mult(*(args[:i]+[derivative(t, x)]+args[i+1:]))
        return result
    
    
def _prod(l):
    c = 1
    for e in l:
        c *= e
    return c

def _distribute_list(lista):
    """
    recibe una lista con los elementos de cada suma que haya en una expresion 
    y distribuye los elementos con todos los elementos 
    """
    k=0
    new_lista=[lista[0]]
    while k < len(lista)-1 : 
        aux=[]
        for i in range(len(new_lista[k])):
            for j in range(len(lista[k+1])):
                aux.append(mult(new_lista[k][i],lista[k+1][j]))
        new_lista.append(aux)
        k +=1 
    return new_lista[-1]


def expand(exp):
    try:
        return exp.expanded()
    except AttributeError:
        return exp
    
class fraction (expr):
 
    def __new__(cls, numer, denom):
        if denom == const(0) or denom == 0:
            raise ZeroDivisionError('Division by zero')
        instance = super(fraction, cls).__new__(cls)
        instance.numer = numer
        instance.denom = denom
        instance._simplify()
        if instance.denom == 1 or instance.denom == const(1):
            return instance.numer
        else:
            return instance

 
    def __init__(self, numer = const(1), denom = const(1)):
        self.args = (self.numer, self.denom)
        self._mhash = None
     
    def _simplify(self):
        n, new_numer = _separate_const(self.numer)
        d, new_denom = _separate_const(self.denom)
        if n.is_integer() and d.is_integer():
            gcd = _gcd(n.val(), d.val())
            self.numer = const(int(n.val()/gcd))*new_numer
            self.denom = const(int(d.val()/gcd))*new_denom
    
    def __str__(self):
        if self.denom == const(1):
            return '%s'%(str(self.numer))
        elif isinstance(self.denom, mult):
            return '('+'%s/(%s)'%(str(self.numer),str(self.denom))+')'
        else:
            return '('+'%s/%s'%(str(self.numer),str(self.denom))+')'
     
    def latex(self):
        if self.denom == const(1):
            return '%s'%(self.numer.latex())
        else:
            return '\\frac{%s}{%s}'%(self.numer.latex(),self.denom.latex())
     
    def __add__(self, other):
        if _isint(self.numer) and _isint(self.denom):
            if isinstance(other, fraction):
                if _isint(other.numer) and _isint(other.denom):
                    return fraction(self.numer*other.denom \
                                    + self.denom*other.numer, 
                                    other.denom*self.denom)
            elif _isint(other):
                return fraction(self.numer + other.val()*self.denom, 
                                self.denom)
        
        return plus(self,other)
     
    def __radd__(self, other):
        return self.__add__(other)
     
    def __sub__(self, other):
        return self + const(-1)*other
     
    def __rsub__(self, other):
        return const(-1)*self + other
     
    def __mul__(self, other):
        if isinstance(other, fraction):
            if _isint(other.numer) and _isint(other.denom):
                return fraction(self.numer*other.numer, self.denom*other.denom)
        elif _isint(other):
            return fraction(self.numer*other, self.denom)
        elif isinstance(other, int) or isinstance(other, float):
            return mult(const(other), self)
    
        return mult(self,other)
     
    def __rmul__(self, other):
        return self.__mul__(other)
     
    def __truediv__(self, other):
        if isinstance(other, fraction):
            if _isint(other.numer) and _isint(other.denom):
                return fraction(self.numer*other.denom, self.demon*other.numer)
        elif _isint(other, const):
            return fraction(self.numer, self.denom*other)
    
        return fraction(self, other)
     
    def __rtruediv__(self, other):
        if isinstance(other, fraction):
            if _isint(other.numer) and _isint(other.denom):
                return fraction(other.numer*self.denom, other.denom*self.numer)
        elif _isint(other):
            return fraction(self.denom*other, self.numer)
    
        return fraction(other, self)
    
    def expanded(self):
    
        new_numer = expand(self.numer)
        new_denom = expand(self.denom)
    
        if not isinstance(new_numer, plus):
            return fraction(new_numer, new_denom)
        
        terms = list(i/new_denom for i in new_numer.args)
        return plus(*terms)
    
    def _separate_const(self):
        cn, tn = _separate_const(self.numer)
        cd, td = _separate_const(self.denom)
        return const_fraction(cn, cd), fraction(tn, td)
    
    def deriv(self, x):
        return const(-1)*(self.numer/self.denom**2)*derivative(self.denom,x) +\
            derivative(self.numer,x)/self.denom
        
    
def _gcd(a, b):
    while b:
        a, b = b, a%b
    return a

def _isint(c):
    if isinstance(c, const):
        return c.is_integer()
    else:
        return False
 
 
def _is_number(e):
    return isinstance(e, const) or isinstance(e, float) or isinstance(e, int)
   
def _power_nums(base, exponent):
    
    b = base.val() if isinstance(base, const) else base
    e = exponent.val() if isinstance(exponent, const) else exponent
    return const(b**e)
        
    
class power (expr):
  
    def __new__(cls, base, exponent):
        
        instance = super(power, cls).__new__(cls)
        if exponent == 1 or exponent == const(1):
            return base
        elif isinstance(base, power):
            return power(base.base, base.exponent + exponent)
        elif _is_number(base) and _is_number(exponent):
            return _power_nums(base, exponent)
        else:
            return instance
    
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
        self.args = (base, exponent)
        self._mhash = None
    
    def __str__(self):
        base_string = str(self.base)
        exponent_string = str(self.exponent)
        if isinstance(self.base, mult):
            base_string = '(' + base_string + ')'
        if isinstance(self.exponent, mult):
            exponent_string = '(' + base_string + ')'
        return '%s^%s'%(base_string, exponent_string)
    
    def latex(self):
        base_string = str(self.base)
        if isinstance(self.base, mult):
            base_string = '(' + base_string + ')'
        return '%s^{%s}'%(base_string, str(self.exponent))
    
    def expanded(self):
        
        if isinstance(self.exponent, const):
            integer_exponent = self.exponent.is_integer()
            val = self.exponent.val()
        else:
            integer_exponent = isinstance(self.exponent, int)
            val = self.exponent
        
        if integer_exponent:
            if isinstance(self.base, plus):
                split = list(self.base.args)
            else:
                split = [self.base]
            terms = [split]*val
            return sum(_distribute_list(terms))
        else:
            return self
        
    def deriv(self, x):
        if derivative(self.exponent, x) != const(0):
            import warnings
            warnings.warn('Derivative of exponents not implemented.')
        return self.exponent*power(self.base, self.exponent - 1)*\
            derivative(self.base, x)
        
        
def derivative(exp, var):
    try:
        return exp.deriv(var)
    except AttributeError:
        return const(0)
    

if __name__ == '__main__':
    x = var('x')
    y = var('y')
    print(derivative(x*y/(x+1), x))