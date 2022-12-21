# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 07:23:49 2021

@author: jorgenorena
based on code written by Joaquin Rohland
"""

import sympy as sp
import numpy as np
import numbers

symbol_dict = {}

def vectors(*names):
    return tuple(Vector(name) for name in names)

def scalars(*names):
    return tuple(Scalar(name) for name in names)

def srepr(a):
    try:
        return a.srepr()
    except AttributeError:
        return repr(a)

def latex(a):
    try:
        return a.latex()
    except AttributeError:
        return str(a)

def code(a):
    try:
        return a.code()
    except AttributeError:
        return str(a)

def _gcd(a, b):
    while b:
        a, b = b, a%b
    return a

def prod(args):
    res = 1
    for a in args:
        res *= a
    return res

def is_number(n):
    return isinstance(n, numbers.Number)

def is_not_number(n):
    return not isinstance(n, numbers.Number)

def is_vector(expr):
    try:
        return expr.vector
    except AttributeError:
        return False

def is_scalar(expr):
    try:
        return expr.scalar
    except AttributeError:
        if isinstance(expr, numbers.Number):
            return True
        else:
            return False

class VExpr():
    
    def __init__(self):
        self._mhash = None
    
    
    def __hash__(self):
        h = self._mhash 
        if h is None:
            h = hash((type (self).__name__,) + \
                     tuple (self.args))
            self._mhash = h
        return h
        
    
    def srepr(self):
        return type(self).__name__ + '(' +\
            ', '.join(srepr(a) for a in self.getargs()) + ')'
            
    def __repr__(self):
        return self.srepr()
    
    def __add__(self, other):
        
        if other == 0:
            return self
        
        if is_vector(self):
            if is_vector(other):
                return VectSum(self, other)
            else:
                raise TypeError("unsupported operand type(s) for +: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
            
        elif is_scalar(self):
            if is_scalar(other):
                return ScalSum(self, other)
            elif is_series(other):
                return other + self
            else:
                raise TypeError("unsupported operand type(s) for +: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)

        
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-1)*other

    def __rsub__(self, other):
        return other + (-1)*self

    def __mul__(self, other):
        if is_vector(self):
            if is_scalar(other):
                return VectScalMul(other, self)
            elif is_vector(other):
                return Dot(self, other)
            elif is_series(other):
                return other*self
            else:
                raise TypeError("unsupported operand type(s) for *: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
                
        elif is_scalar(self):
            if is_scalar(other):
                return ScalMul(self, other)
            elif is_vector(other):
                return VectScalMul(self, other)
            elif is_series(other):
                return other*self
            else:
                raise TypeError("unsupported operand type(s) for *: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
        
        
    def __rmul__(self, other):
        return self * other
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __truediv__(self, other):
        if is_vector(self):
            if is_scalar(other):
                return VectScalMul(ScalPow(other, -1), self)
            else:
                return TypeError("unsupported operand type(s) for /: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
        
        elif is_scalar(self):
            if is_scalar(other):
                return ScalMul(ScalPow(other, -1), self)
            else:
                raise TypeError("unsupported operand type(s) for /: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
        
        else:
            raise TypeError("unsupported operand type(s) for /: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
        
    def __rtruediv__(self, other):
        if is_scalar(self):
            if is_vector(other):
                return VectScalMul(ScalPow(self, -1), other)
            elif is_scalar(other):
                return ScalMul(ScalPow(self, -1), other)
            else:
                raise TypeError("unsupported operand type(s) for /: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)
        else:
            raise TypeError("unsupported operand type(s) for /: " +\
                                type(self).__name__ + ' and ' +\
                                    type(other).__name__)              

    def __pow__(self, other):
        if is_scalar(self) and is_scalar(other):
            return ScalPow(self, other)
        else:
            print("self: ", type(self))
            print("other: ", type(other))
            raise TypeError("Power only supported among scalars")

    def __neg__(self):
        if is_vector(self):
            return VectScalMul(-1, self)
        elif is_scalar(self):
            return ScalMul(-1, self)
        else:
            raise TypeError("Negative only supported for scalars or vectos.")

    def getargs(self):
        return self.args
 
class Associative():   
    
    def make_associative(self):    
        new_args = []
        for a in self.args:
            if type(a) == type(self):
                new_args.extend(a.args)
            else:
                new_args.append(a)
        self.args = tuple(new_args)
        
    
class Commutative():  

    def make_commutative(self):
        constlist = list(filter(is_number, self.args))
        arglist = sorted(list(filter(is_not_number, self.args)), key=hash)
        if len(constlist) > 0:
            arglist = [self._number_version(*constlist)] + arglist
        self.args = tuple(arglist)


class Identity():
    
    def ignore_identity(self):
        def not_equal_to_identity(x):
            return x != self._identity_
        self.args = tuple(filter(not_equal_to_identity, self.args))
        if len(self.args) == 0:
            self.args = (self._identity_,)
    

class NullElement():
    
    def is_null(self):
        return self._null_ in self.args
    
    
class Cummulative():
    
    def simplify(self, repeated, operate, separate):
        previous = None
        c = None
        terms = []
        def key(term):
            ci, t  = separate(term)
            return hash(t)
        args = sorted(self.args, key=key)
        for term in args:
            ci, current = separate(term)
            if current != previous:
                if previous != None:
                    terms.append(repeated(previous, c))
                c = ci
                previous = current
            else:
                c = operate(c, ci)
        terms.append(repeated(previous, c))
        self.args = tuple(terms)
    
    
class Vector(VExpr):
    
    def __init__(self, name, components = None):
        if not isinstance(name, str):
            raise ValueError('Name of vector must be a string.')
        self.name = name
        self.components = components
        self.vector = True
        self.args = (name,)
        self._mhash = None
    
    def getargs(self):
        return self.args
    
    def latex(self):
        return '\\vec{' + self.name + '}'
    
    def __repr__(self):
        return 'v('+self.name+')'
    
    def code(self):
        return self.name

    def srepr(self):
        return repr(self)
    
    def set_components(self, components):
        self.components = components
        
    def val(self):
        if self.components != None:
            return self.components
        else:
            return self
    
class VectSum(VExpr, Associative, Commutative, Identity, Cummulative):
    
    def __new__(cls, *args):
        if not all(map(is_vector, args)):
            raise TypeError('VectSum should only involve vector objects.')
        instance = super(VectSum, cls).__new__(cls)
        instance._identity_ = Vector('0')
        instance.args = args
        instance.make_associative()
        instance.make_commutative()
        instance.ignore_identity()
        instance.simplify(lambda a, b: VectScalMul(b, a), ScalSum, 
                          instance._separate_scal)
        if len(instance.args) == 1:
            return instance.args[0]
        else:
            return instance
    
    def __init__(self, *args):
        self.vector = True
        self._mhash = None 

    def _separate_scal(self, term):
        if isinstance(term, VectScalMul) and is_scalar(term.args[0]):
            return term.args[0], term.args[1]
        else:
            return 1, term

    def __repr__(self):
        return '(' + ' + '.join(repr(a) for a in self.args) + ')'
        
    def latex(self):
        l = [latex(a) for a in self.args]
        return '(' + ' + '.join(l) + ')'
    
    def val(self):
        return sum(a.val() for a in self.args)
    

class VectScalMul(VExpr, Identity):
    
    def __new__(cls, *args):
        if len(args) != 2:
            raise TypeError('VectScalMul takes 1 argument, ' + len(args) +\
                            'were given')
        elif not is_scalar(args[0]) or not is_vector(args[1]):
            raise TypeError('VectScalMul takes a scalar and a vector.')
        
        instance = super(VectScalMul, cls).__new__(cls)
        instance.args = args
        instance._identity_ = 1
        instance.ignore_identity()
        if len(instance.args) == 1:
            return instance.args[0]
        elif instance.args[0] == 0 or instance.args[1] == Vector('0'):
            return Vector('0')
        elif isinstance(instance.args[1], VectSum):
            return VectSum(*[instance.args[0]*v 
                             for v in instance.args[1].args])
        else:
            return instance
        
    def __init__(self, *args):
        self.vector = True
        self._mhash = None
        
    def __repr__(self):
        return repr(self.args[0]) + '*' + repr(self.args[1])
    
    def latex(self):
        return latex(self.args[0]) + ' ' + latex(self.args[1])
    
    def val(self):
        return self.args[0].val()*self.args[1].val()


class Dot(VExpr, Commutative, NullElement):
    
    def __new__(cls, *args):
        if len(args) != 2:
            raise TypeError('Dot takes 1 argument, ' + len(args) +\
                            'were given')
        if not all(map(is_vector, args)):
            raise TypeError('Dot should only involve vector objects.')
        instance = super(Dot, cls).__new__(cls)
        instance._null_ = Vector('0')
        instance.args = args
        instance.make_commutative()
        if instance.is_null():
            return 0
        # if there are sums or products with scalars, expand
        if isinstance(instance.args[0], VectSum) \
            or isinstance(instance.args[1], VectSum):
            return instance.expand_sum()
        elif isinstance(instance.args[0], VectScalMul) \
            or isinstance(instance.args[1], VectScalMul):
            return instance.expand_mul()
        else:
            return instance
     
    def expand_sum(self):
        if isinstance(self.args[0], VectSum):
            terms0 = self.args[0].args
        else:
            terms0 = [self.args[0]]
        
        if isinstance(self.args[1], VectSum):
            terms1 = self.args[1].args
        else:
            terms1 = [self.args[1]]
            
        sumargs = [Dot(t1, t2) for t1 in terms0 for t2 in terms1]
        return ScalSum(*sumargs)
    
    
    def expand_mul(self):
        if isinstance(self.args[0], VectScalMul):
            s1, t1 = self.args[0].args
        else:
            s1 = 1
            t1 = self.args[0]
        
        if isinstance(self.args[1], VectScalMul):
            s2, t2 = self.args[1].args
        else:
            s2 = 1
            t2 = self.args[1]
        
        return ScalMul((s1*s2),Dot(t1, t2))
    
        
    def __init__(self, *args):
        self.scalar = True
        self._mhash = None
        self.symbol = None
        
    def __repr__(self):
        return '(' + repr(self.args[0]) + '.' + repr(self.args[1]) + ')'
    
    def latex(self):
        return latex(self.args[0]) + ' \cdot ' + latex(self.args[1])
    
    def val(self):
        return sum(i[0]*i[1] 
                   for i in zip(self.args[0].val(), self.args[1].val()))
    
    def code(self):
        return code(self.args[0]) + 'd' + code(self.args[1])

    def sympy(self, output_type):
        if self.symbol == None:
            self.make_symbol(output_type)
        return self.symbol
    
    def make_symbol(self, output_type):
        if output_type == "latex":
            self.symbol = sp.Symbol(self.latex())
        elif output_type == "code":
            self.symbol = sp.Symbol(self.code())
        symbol_dict[self.symbol] = self
        
    def clean_symbol(self):
        self.symbol = None
        del symbol_dict[self.symbol]
    

    
class Scalar(VExpr):
    
    def __init__(self, name, value = None):
        if not isinstance(name, str):
            raise ValueError('Name of scalar must be a string.')
        self.name = name
        self.value = value
        self.scalar = True
        self.symbol = None
        self.args = (name,)
        self._mhash = None
        
    def getargs(self):
        return (self.name,)
    
    def __repr__(self):
        return self.name
        
    def latex(self):
       return self.name
   
    def srepr(self):
        return repr(self)
   
    def set_value(self, value):
        self.value = value
   
    def val(self):
        if self.value != None:
            return self.value
        else:
            return self
    
    def sympy(self, output_type):
        if self.symbol == None:
            self.make_symbol()
        return self.symbol
    
    def make_symbol(self):
        self.symbol = sp.symbols(self.name)
        symbol_dict[self.symbol] = self
        
    def clean_symbol(self):
        self.symbol = None
        del symbol_dict[self.symbol]


class ScalPow(VExpr):
    
    def __new__(cls, *args):
        if not all(map(is_scalar, args)):
            raise TypeError('ScalPow should only involve Scalar objects.')
        instance = super(ScalPow, cls).__new__(cls)
        instance.args = args
        if len(instance.args) > 2:
            raise TypeError('ScalPow takes 2 arguments but ' + \
                            len(instance.args)  + ' were given.')
        elif len(instance.args) == 1:
            return instance.args[0]
        elif instance.args[0] == 1:
            return 1
        elif instance.args[1] == 0:
            return 1
        elif instance.args[1] == 1:
            return instance.args[0]
        elif isinstance(instance.args[0], float) and \
            is_number(instance.args[1]):
            return instance.args[0]**instance.args[1]
        elif is_number(instance.args[0]) and \
            isinstance(instance.args[1], float):
            return instance.args[0]**instance.args[1]
        else:
            return instance
        
    def __init__(self, *args):
        self._mhash = None
        self.scalar = True
        self.base = self.args[0]
        self.exp = self.args[1]
        
    def __repr__(self):
        
        base_string = repr(self.base)
        
        if is_number(self.exp) and self.exp < 0:
            if self.exp == -1:
                return '(1/' + base_string + ')'
            exp_string = repr(-self.exp)
            return '(1/(' + base_string + '^' + exp_string + '))'
        else:
            exp_string = repr(self.exp)
            return '(' + base_string + ')^' + exp_string
        
    def latex(self):
        
        base_string = latex(self.base)
        
        if is_number(self.exp) and self.exp < 0:
            exp_string = str(-self.exp)
            return '\\frac{1}{' + base_string + '^' + exp_string + '}'
        else:
            exp_string = latex(self.exp)
            return '(' + base_string + ')^' + exp_string

    def val(self):
        return self.base.val()**self.exp.val()
    
    def sympy(self, output_type):
        return sp.Pow(sympy(self.base, output_type), 
                      sympy(self.exp, output_type))


class ScalMul(VExpr, Associative, Commutative, Identity, Cummulative, 
              NullElement):
    
    def __new__(cls, *args):
        if not all(map(is_scalar, args)):
            raise TypeError('ScalMul should only involve Scalar objects.')
        instance = super(ScalMul, cls).__new__(cls)
        instance._identity_ = 1
        instance._null_ = 0
        instance.args = args
        instance.make_associative()
        instance.make_commutative()
        if instance.is_null():
            return 0
        instance.simplify(ScalPow, lambda a, b: a + b,
                          instance._separate_exp)
        instance.ignore_identity()
        if len(instance.args) == 1:
            return instance.args[0]
        elif all([is_number(a) for a in instance.args]):
            return prod(instance.args)
        else:
            return instance
        
    def __init__(self, *args):
        self.scalar = True
        self._mhash = None
        
    def __repr__(self):
        s = [self._separate_exp(a) for a in self.args]
        numer = ''
        denom = ''
        for p, b in s:
            b_repr = repr(b)
            if is_number(p) and p < 0:
                if p == -1:
                    denom += b_repr
                else:
                    p_str = repr(-p)
                    denom += ' ' + b_repr + '^' + p_str
            else:
                if p == 1:
                    numer += ' ' + b_repr
                else:
                    p_str = repr(p) if is_number(p) else repr(p)
                    numer += ' ' + b_repr + '^' + p_str
        if len(numer) == 0:
            numer = str(1)
        else:
            numer = numer[1:]
        if len(denom) == 0:
            return numer
        else:
            return '(' + numer + ')/(' + denom + ')'
    
    def latex(self):
        s = [self._separate_exp(a) for a in self.args]
        numer = ''
        denom = ''
        for p, b in s:
            b_str = latex(b)
            if is_number(p) and p < 0:
                if p == -1:
                    denom += b_str
                else:
                    p_str = str(-p)
                    denom += ' ' + b_str + '^' + p_str
            else:
                if p == 1:
                    numer += ' ' + b_str
                else:
                    p_str = latex(p)
                    numer += ' ' + b_str + '^' + p_str
        if len(numer) == 0:
            numer = str(1)
        else:
            numer = numer[1:]
        if len(denom) == 0:
            return numer
        else:
            return '\\frac{' + numer + '}{' + denom + '}'
    
    def val(self):
        return prod([a.val() for a in self.args])
    
    def sympy(self, output_type):
        return sp.Mul(*[sympy(a, output_type) for a in self.args])

    def _number_version(self, *args):
        return prod(args)
    
    def _separate_exp(self, term):
        if isinstance(term, ScalPow) and is_number(term.args[1]):
            return term.args[1], term.args[0]
        else:
            return 1, term

class ScalSum(VExpr, Associative, Commutative, Identity, Cummulative):
    
    def __new__(cls, *args):
        if not all(map(is_scalar, args)):
            raise TypeError('ScalSum should only involve Scalar objects.')
        instance = super(ScalSum, cls).__new__(cls)
        instance._identity_ = 0
        instance.args = args
        instance.make_associative()
        instance.make_commutative()
        instance.ignore_identity()
        instance.simplify(ScalMul, instance._number_version, 
                          instance._separate_num)
        if len(instance.args) == 1:
            return instance.args[0]
        if all([is_number(a) for a in instance.args]):
            return sum(args)
        else:
            return instance
    
    def __init__(self, *args):
        self.scalar = True
        self._mhash = None 

    def _separate_num(self, term):
        if isinstance(term, ScalMul) and is_number(term.args[0]):
            return term.args[0], ScalMul(*term.args[1:])
        else:
            return 1, term
        
    def __repr__(self):
        l = [(str(a) if is_number(a) else repr(a)) for a in self.args]
        return '(' + ' + '.join(l) + ')'
        
    def latex(self):
        l = [latex(a) for a in self.args]
        return '(' + ' + '.join(l) + ')'

    def val(self):
        return sum(a.val() for a in self.args)
    
    def sympy(self, output_type):
        return sp.Add(*[sympy(a, output_type) for a in self.args])
    
    def _number_version(self, *args):
        return sum(args)


def is_series(expr):
    try:
        return expr.is_series
    except AttributeError:
        return False

class Series:

    ## TO DO:
    # - series inside scalar and vector expressions
    # - symbolic coefficients

    def __init__(self, expansion_parameters, terms):
        self.terms = terms
        self.expansion = expansion_parameters
        self.is_series = True

    def __mul__(self, other):
        if is_series(other) and self.expansion == other.expansion:
            new_terms = []
            for o in range(min(len(self.terms), len(other.terms))):
                result = 0
                for i in range(o + 1):
                    result += self.terms[i]*other.terms[o - i]
                new_terms.append(result)
        else:
            new_terms = [other*x for x in self.terms]

        return Series(self.expansion, new_terms)

    def __rmul__(self, other):
        return self*other

    def __add__(self, other):
        if is_series(other) and self.expansion == other.expansion:
            new_terms = [self.terms[i] + other.terms[i] 
                        for i in range(min(len(self.terms), len(other.terms)))
                        ]
        else:
            new_terms = self.terms[:]
            new_terms[0] += other
        
        return Series(self.expansion, new_terms)

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        
        if not is_scalar(other):
            raise TypeError("Non-scalar powers of series not supported")
        
        max_order = len(self.terms) - 1
        new_terms = []
        power = other

        # Pre-computing combinatorics
        factors = [1]       
        for i in range(max_order + 1):
            factors.append(factors[i]*(power - i)/(i + 1))

        # Ways in which each integer can be written as a sum of smaller integers
        def partition(n):
            if n == 0:
                result = np.zeros(max_order, dtype=int)
                yield result
            for i in range(n,  0, -1):
                result = np.zeros(max_order, dtype=int)
                result[i - 1] = 1
                for p in partition(n - i):
                    if not any(p[i:]):
                        yield result + p

        # We build the final expression order by order
        # o is the order
        for o in range(max_order + 1):
            term = 0
            for part in partition(o):
                result = 1
                result *= self.terms[0]**(power - o)
                for i, p in enumerate(part):
                    result *= factors[p]*self.terms[i+1]**p
                term += result
            new_terms.append(term)

        return Series(self.expansion, new_terms)

    def __str__(self):
        result = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                result += str(term) + ' + '
            elif term != 0:
                result += str(term) + '*' + str(self.expansion) + '^' + str(i) \
                    + ' + '
        result += 'O(' + str(self.expansion) + '^' + str(len(self.terms)) + ')'
        return result

    def __repr__(self):
        result = "Series("
        result += "expansion = " + str(self.expansion) + ", "
        result += "terms = " + str(self.terms) + ")"
        return result

    def expand(self):
        if not all(map(is_scalar, self.terms)):
            raise TypeError("Expansion not supported for non-scalar series.")
        
        expanded = [ScalMul(ScalPow(self.expansion, i), term) for i, term in enumerate(self.terms)]

        return ScalSum(*expanded)

    def sympy(self, output_type):
        return sympy(self.expand(), output_type)


def fract(numer, denom):
    if (isinstance(numer, float) and is_number(denom)) or\
        (isinstance(denom, float) and is_number(numer)):
        return numer/denom
    elif not is_scalar(numer) or not is_scalar(denom):
        raise TypeError('Can only divide scalars (or numbers).')
    else:
        return ScalMul(numer, ScalPow(denom, -1))

def sympy(expr, output_type = "latex"):
    try:
        return expr.sympy(output_type)
    except AttributeError:
        return expr

def is_sympy_number(tree):
    return isinstance(tree, sp.Float) or isinstance(tree, sp.Integer)

def translate_sympy(tree):
    if isinstance(tree, sp.Add):
        return(ScalSum(*[translate_sympy(a) for a in tree.args]))
    elif isinstance(tree, sp.Mul):
        return(ScalMul(*[translate_sympy(a) for a in tree.args]))
    elif isinstance(tree, sp.Pow):
        return ScalPow(*[translate_sympy(a) for a in tree.args])
    elif isinstance(tree, sp.Rational):
        return fract(translate_sympy(tree.p), translate_sympy(tree.q))
    elif isinstance(tree, sp.Symbol) or is_sympy_number(tree):
        return symbol_dict[tree]
    elif is_number(tree):
        return tree
    else:
        print('Could not translate:', tree)
    
if __name__ == '__main__':
    
    r = ScalMul(2, 1/2)
    print(r)
# %%
#repr(x*s**ScalMul(1,ScalPow(2,-1)))

# %%
