from kanren import run, var, fact
import kanren.assoccomm as la

# define mathematical operations
add = 'addition'
mul = 'multiplication'

# declare that these operations are commutative using the facts system
fact(la.commutative, mul)
fact(la.commutative, add)
fact(la.associative, mul)
fact(la.associative, add)

# define some variables
a, b, c = var('a'), var('b'), var('c')

# Consider the following expression: expression_orig = 3 x (-2) + (1 + 2 x 3) x (-1)

# generate expressions
expression_orig = (add, (mul, 3, -2), (mul, (add, 1, (mul, 2, 3)), -1))
expression1 = (add, (mul, (add, 1, (mul, 2, a)), b), (mul, 3, c))
expression2 = (add, (mul, c, 3), (mul, b, (add, (mul, 2, a), 1)))
expression3 = (add, (add, (mul, (mul, 2, a), b), b), (mul, 3, c))

# compare expressions (1st = number of values, 2nd = variable, 3rd = function)
print(run(0, (a, b, c), la.eq_assoccomm(expression1, expression_orig)))
print(run(0, (a, b, c), la.eq_assoccomm(expression2, expression_orig)))
print(run(0, (a, b, c), la.eq_assoccomm(expression3, expression_orig)))

