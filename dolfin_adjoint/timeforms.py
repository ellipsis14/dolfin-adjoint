import dolfin
import ufl 

class TimeConstant(object):
    def __init__(self, label):
        self.label = label
    def __repr__(self):
        return 'TimeConstant("'+self.label+'")'

START_TIME = TimeConstant("START_TIME")
FINISH_TIME = TimeConstant("FINISH_TIME")

def timeslice(inslice):
    '''Preprocess a time slice, replacing the start with START_TIME and
    stop with FINISH_TIME.'''
    if not isinstance(inslice, slice):
        return inslice

    if inslice.start is None:
        start = START_TIME
    else:
        start = inslice.start
        
    if slice.stop is None:
        stop = FINISH_TIME
    else:
        stop = inslice.stop

    return slice(start, stop, inslice.step)


class TimeTerm(object):
    '''A form evaluated at a point in time or over a time integral.'''
    def __init__(self, form, time):
        self.form = form
        self.time = time

    def __repr__(self):
        return "TimeTerm("+self.form.__repr__()+",time = "+\
            repr(self.time)+")"    

    def __neg__(self):
        return TimeTerm(-self.form,self.time)


class TimeForm(object):
    def __init__(self, terms):
        try:
            self.terms = list(terms)
        except TypeError:
            self.terms = [terms]

    def __add__(self, other):
        # Adding occurs by concatenating terms in the forms list.
        
        if isinstance(other, TimeForm):
            sum = TimeForm()        
            sum.terms = self.terms + other.terms
            return sum

        else:
            return NotImplemented

    def __sub__(self, other):
        # Subtract by adding the negation of all the terms.
        
        if isinstance(other, TimeForm):
            sum = TimeForm()        
            sum.terms = self.terms + [-term for term in other.terms]
            return sum

        else:
            return NotImplemented

    def __neg__(self):
        # Unary negation occurs by negating the terms.

        neg = TimeForm()
        neg.terms = [-term for term in other.terms]
        
    def __repr__(self):
        return "TimeForm("+repr(self.terms)+")"

def at_time(form, time):
    '''Form a TimeForm evaluated at a particular time point from a form and
    a time.'''
    
    return TimeForm(TimeTerm(form, time))


class TimeMeasure(object):
    '''Define a measure for an integral over some interval in time.'''
    def __init__(self, interval = slice(START_TIME,FINISH_TIME,None)):

        self.interval = timeslice(interval)

    def __getitem__(object, key):
        
        return TimeMeasure(timeslice)

    def __rmul__(self, other):
        
        if isinstance(other, ufl.form.Form):
            # Multiplication with a form produces the TimeForm.
            return TimeForm(TimeTerm(other, self.interval))
            
        else:
            return NotImplemented

    def __repr__(self):
        return "TimeMeasure(interval = "+repr(self.interval)+")"

dt = TimeMeasure()

if __name__ == "__main__":
    from dolfin import *
    
    mesh = UnitSquare(2,2)
    
    U = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(U)

    F = v*dx
    
    TF = F*dt
    
    AT = at_time(F,0.0) 
