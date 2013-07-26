class FixedLineSearch:
    def __init__(self, start_stp = 1.0):
        '''
        This class implements a dummy line search algorithm whose steps 
        are simply the initial guess.
        '''

        self.start_stp    = start_stp

    def search(self, phi, dphi):
        ''' Performs the dummy line search on the function phi. 

            The return value is the starting step. 
        '''
        return self.start_stp   
