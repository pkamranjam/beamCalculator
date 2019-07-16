import numpy as np
import scipy.integrate
import itertools
import sys
import json


class OutOfBeamException(Exception):
    """
    This exception is raised when a point load or part of a distributesd load
    is outside a defined beam.
    """

class ID(object):
    """
    An ID represents a custom id for each element
    """
    Id = 1
    """
    Initializes a new id for an element and increments the class
    variable Id by one
    """
    def __init__(self):
        self.Id = ID.Id
        ID.Id += 1
        
    def getId(self):
        return self.Id


class Point(object):
    """
    A point represents a location in space
    """
    def __init__(self, x, y, z=0):
        """
        Initializes x, y and z coordinate of the point object.
        x, y, z: integer or float
        """
        if not(type(x)==int or type(x)==float):
            raise TypeError ('X coordinate of point has to be an integer or \
                             a floating point number.')
            
        if not(type(y)==int or type(y)==float):
            raise TypeError ('Y coordinate of point has to be an integer or \
                             a floating point number.')
            
        if not(type(z)==int or type(z)==float):
            raise TypeError ('Z coordinate of point has to be an integer or \
                             a floating point number.')
            
        self.point = [x, y, z]
        
    def getX(self):
        return self.point[0]
    
    def getY(self):
        return self.point[1]
    
    def getZ(self):
        return self.point[2]
    
    def getPoint(self):
        return self.point
    
    def translatePoint(self, vector):
        """
        Translates a point in space using a given vector. 
        vector: an instance of class Vector.
        Returns: Point object (an instance of Point class)
        """
        if not isinstance(vector, Vector):
            raise TypeError ('vector must be an instance of Vector class.')
        return Point(self.getX()+vector.getX(), self.getY()+vector.getY(),\
                     self.getZ()+vector.getZ())
        
    def distanceFromPoint(self, point):
        """
        Calculates the distance between two points, with this object as the 
        current point and point as the other given point.
        point: an instance of Point class.
        Returns: euclidean distance between two points.
        """
        if not isinstance(point, Point):
            raise TypeError ('point must be an instance of the Point class')
        return ((self.getX()-point.getX())**2+(self.getY()-point.getY())**2+ \
                       (self.getZ()-point.getZ())**2)**0.5


class Vector(object):
    """
    Represents a vector in the euclidean space.
    """
    def __init__(self, x, y, z=0):
        """
        Initializes x, y and z coordinates of a vector.
        x, y, z: integer or float
        """
        if not(type(x)==int or type(x)==float):
            raise TypeError ('X coordinate of vector has to be an integer or \
                             a floating point number.')
            
        if not(type(y)==int or type(y)==float):
            raise TypeError ('Y coordinate of vector has to be an integer or \
                             a floating point number.')
            
        if not(type(z)==int or type(z)==float):
            raise TypeError ('Z coordinate of vector has to be an integer or \
                             a floating point number.')
            
        self.vector = [x, y, z]
        
    def getX(self):
        return self.vector[0]
    
    def getY(self):
        return self.vector[1]
    
    def getZ(self):
        return self.vector[2]
    
    def getVector(self):
        return self.vector
    
    def vectorMagnitude(self):
        """
        Returns vector's magnitude or length.
        """
        return np.linalg.norm(self.vector)
    
    def vectorAddition(self, other):
        """
        Adds two vectors coordinate-wise in euclidean space, with this object as
        the current vector and other as a given vector.
        other: an instance of Vector class
        Returns: an instance of Vector class
        """
        if not isinstance(other, Vector):
            raise TypeError ('other must be an instance of Vector class.')
        return Vector(self.getX()+other.getX(), self.getY()+other.getY(), \
                      self.getZ()+other.getZ())

class PointLoad(object):
    """
    Represents a point load in space.
    """
    def __init__(self, positionPoint, load, angle=90):
        """
        Initializes a point load with a given position point, magnitude and angle.
        It decomposes load vector into two vertical (loadVectorY) and horizontal
        (loadVectorX) vectors in order to calculate the sum of forces in x and 
        y directions. Finally, it evaluates a polynomial (a constant) resulting 
        from the effect of point load in an inteval of a beam.
        
        positionPoint: An instance of Point class.
        Load: int or fload representing the magnitude of point load
        angle: int or float representing the angle between point load and beam
                ranging from 0 to 360.
        """
        if not isinstance(positionPoint, Point):
            raise TypeError ('positionPoint must be an instance of Point class.')
            
        if not(type(load)==int or type(load)==float):
            raise TypeError ('Load must be an integer or \
                             a floating point number.')
            
        if not(type(angle)==int or type(angle)==float):
            raise TypeError ('Angle must be an integer or \
                             a floating point number.')
            
        if angle<0 or angle>180:
            raise ValueError ('Angle must be less than 180 and greater than zero.')
            
        self.Id = ID()
        self.positionPoint = positionPoint
        self.load = load
        self.angle = angle
        self.loadVectorY = Vector(0, float(self.load*np.sin(np.radians(self.angle))), 0)
        self.loadVectorX = Vector(float(self.load*np.cos(np.radians(self.angle))), 0, 0)
        self.polynomial = np.poly1d([self.loadVectorY.getY()])
        
    def getId(self):
        return self.Id.getId()
    
    def getPositionPoint(self):
        return self.positionPoint
    
    def setPositionPoint(self, point):
        """
        Sets a new position point for a point load.
        point: must be an instance of Point class.
        """
        if not isinstance(point, Point):
            raise TypeError ('point must be an instance of Point class.')
        self.positionPoint = point
        
    def getLoad(self):
        return self.load
    
    def getLoadVectorY(self):
        return self.loadVectorY
    
    def getLoadVectorX(self):
        return self.loadVectorX
    
    def getAngle(self):
        return self.angle
    
    def getPolynomial(self):
        return self.polynomial
    
    def isInsideInterval(self, interval):
        """
        Chacks if a point load is inside a given interval excluding the end point.
        The interval is in the x direction (beam axis).
        """
        return self.getPositionPoint().getX() == interval[0]


class DistributedLoad(object):
    """
    Represents a distributed load on a beam.
    """
    def __init__(self, positionPoint1, loadVector1, positionPoint2, loadVector2):
        """
        Initializes a distributed load using two position points representing 
        the start and end points and two loadVectors representing the start and
        end load vectors of a disributed load. It creates the coefficient of a 
        line connecting two loads and polynomial of the line. It calculates the
        area under the line connecting two loads which corresponds to the magnitude
        of the resultant force and also finds the center of gravity, the location
        of the resultant force. Finally, it returns the resultant load as an 
        instance of PointLoad class.
        """
        if not isinstance(positionPoint1, Point):
            raise TypeError ('positionPoint1 must be an instance of Point class.')
            
        if not isinstance(loadVector1, Vector):
            raise TypeError ('loadVector1 must be an instance of Vector class.')
            
        if not isinstance(positionPoint2, Point):
            raise TypeError ('positionPoint2 must be an instance of Point class.')
            
        if not isinstance(loadVector2, Vector):
            raise TypeError ('loadVector2 must be an instance of Vector class.')
            
        if positionPoint1.getX() == positionPoint2.getX():
            raise TypeError ('Start and end loads cannot have the same position point.')
            
        self.Id = ID()
        self.positionPoint1 = positionPoint1
        self.loadVector1 = loadVector1
        self.positionPoint2 = positionPoint2
        self.loadVector2 = loadVector2
        self.coefficients = [(self.loadVector2.getY()-self.loadVector1.getY())/(self.positionPoint2.getX()-self.positionPoint1.getX()),\
                            ((self.loadVector2.getY()-self.loadVector1.getY())/(self.positionPoint2.getX()-self.positionPoint1.getX()))*-self.positionPoint1.getX()+self.loadVector1.getY()]
        
        self.polynomial = np.poly1d(self.coefficients)
#        self.area = np.polysub(np.polyint(self.polynomial), np.poly1d([np.polyint(self.polynomial)(self.positionPoint1.getX())]))
        self.area = scipy.integrate.quad(self.polynomial, self.positionPoint1.getX(), \
                                         self.positionPoint2.getX())[0]
        
        self.centerOfGravity = Point(scipy.integrate.quad(np.polymul(np.poly1d([1,0]), \
                                    self.polynomial), self.positionPoint1.getX(), \
                                    self.positionPoint2.getX())[0]/self.area,0,0)
        
        self.resultantLoad = PointLoad(self.centerOfGravity, self.area, 90)
        
    def getId(self):
        return self.Id.getId()
    
    def getPositionPoint1(self):
        return self.positionPoint1
    
    def getLoadVector1(self):
        return self.loadVector1
    
    def getPositionPoint2(self):
        return self.positionPoint2
    
    def getLoadVector2(self):
        return self.loadVector2
    
    def getCoefficients(self):
        return self.coefficients
    
    def getPolynomial(self):
        return self.polynomial
    
    def getArea(self):
        return self.area
    
    def getCenterOfGravity(self):
        return self.centerOfGravity
    
    def getResultantLoad(self):
        return self.resultantLoad
    
    def isInsideInterval(self, interval):
        """
        Checks if a distributed load is inside a given interval.
        """
        return self.getPositionPoint1().getX() <= interval[0] and self.getPositionPoint2().getX() >= interval[1]
    

class Moment(object):
    """
    Represents a moment on a beam.
    """
    def __init__(self, positionPoint, moment):
        """
        Initializes a moment object using position point and magnitude of moment.
        positionPoint: An instance of Point class.
        moment: int or float. Magnitude of moment.
        """
        if not isinstance(positionPoint, Point):
            raise TypeError ('positionPoint must be an instance of Point class.')
            
        if not (type(moment)==int or type(moment)==float):
            raise TypeError ('moment must be of type int or float.')
            
        self.Id = ID()
        self.positionPoint = positionPoint
        self.moment = moment
        
    def getId(self):
        return self.Id.getId()
    
    def getPositionPoint(self):
        return self.positionPoint
    
    def getMoment(self):
        return self.moment
    
    def isInsideInterval(self, interval):
        return self.getPositionPoint().getX() == interval[0]
    

class Support(object):
    """
    A support represents support of a beam.
    """
    def __init__(self, positionPoint, reacType='fixed'):
        """
        Initializes a support using a given positionPoint and sets the 
        reactionForce and polynomial of the support initially to zero vector 
        and constant respectively.
        
        positionPoint: An instance of Point class.
        reactionForce: An instance of Vector class.
        polynomial: Numpy poly1d polynomial.
        """
        if not isinstance(positionPoint, Point):
            raise TypeError ('positionPoint must be an instance of Point class.')
            
        self.Id = ID()
        self.positionPoint = positionPoint
        self.reactionForce = Vector(0,0,0)
        self.polynomial = np.poly1d([0])
        
    def getId(self):
        return self.Id.getId()
    
    def getPositionPoint(self):
        return self.positionPoint
    
    def getReactionForce(self):
        return self.reactionForce
    
    def setReactionForce(self, force):
        """
        Sets the reaction force and polynomial of a support after calculation.
        "force" must be an instance of Vector class.
        """
        if not isinstance(force, Vector):
            raise TypeError ("force must be an instance of Vector class.")
        self.reactionForce = force
        yVal = self.reactionForce.getY()
        self.polynomial = np.poly1d([yVal])
        
    def getPolynomial(self):
        return self.polynomial
    
    def isToTheRight(self, load):
        """
        Checks if support is to the right side of a given load by comparing their
        x values. 
        
        This method is useful when specifying the sign of moment of 
        load around a given point.
        """
        return self.getPositionPoint().getX() > load.getPositionPoint().getX()
    
    def isInsideInterval(self, interval):
        return self.getPositionPoint().getX() == interval[0]
    

class Beam(object):
    """
    A beam represents a beam defined with a specified length and the defined
    loads, moments and supports acting on it. All the elements acting on a beam
    divide it into different intervals that have specific shear forces and bending
    moments. 
    
    This class calculates these shear forces and bending moments along
    with their equations.
    """
    def __init__(self, pointLoads, distLoads, supports, moments, length):
        """
        Initializes a beam object with given point Loads, distributed loads, 
        supports, moments and beam length.
        
        pointLoads: A list of point loads. All the point loads in this list must 
                    be instances of PointLoad class.
        distLoads: A list of distributed loads. All the distributed loads in 
                    this list must be instances of DistributedLoad class.
        supports: A list of supports. All the supports in this list must be 
                    instances of Support class.
        moments: A list of moments. All the moments in this list must be instances
                 of Moment class.
        length: int or float. Length of beam.
        """
        for load in pointLoads:
            if not isinstance(load, PointLoad):
                raise TypeError ('Load must be an instance PointLoad class.')
                
        for load in distLoads:
            if not isinstance(load, DistributedLoad):
                raise TypeError ('Load must be an instance DistributedLoad class.')
                
        for support in supports:
            if not(isinstance(support, Support)):
                raise TypeError ('Support of the beam must be an instance of Support class.')
                
        for moment in moments:
            if not(isinstance(moment, Moment)):
                raise TypeError ('Moment of the beam must be an instance of Moment class.')
        
        if not(type(length) != int or type(length) != float):
            raise TypeError ("Length must be either int or float.")
                
        self.Id = ID()
        self.pointLoads = pointLoads
        self.distLoads = distLoads
        self.supports = supports
        self.moments = moments
        self.length = length
        self.beam = [elem for array in [self.pointLoads, self.distLoads, self.supports, self.moments] for elem in array]
        self.xVals = self.makeXVals(self.beam)
        self.intervals = self.makeIntervals(self.xVals)
        self.loadIntervals = self.makeLoadIntervals(self.intervals)
        self.shearForces, self.rawSh, self.vi = [], [], []
        self.bendingMoments, self.rawBm, self.mi = [], [], []
        
    def getId(self):
        return self.Id.getId()
    
    def getLoads(self):
        return [elem for array in [self.pointLoads, self.distLoads] for elem in array]
    
    def getSupports(self):
        return self.supports
    
    def getMoments(self):
        return self.moments
    
    def getLength(self):
        return self.length
    
    def getBeam(self):
        return self.beam
    
    def makeXVals(self, beam):
        """
        Creates a list of all x values of elements acting on a beam, including 
        the start and end point of the beam.
        
        Returns: a list of x values.
        """
        xVals = []
        xVals.append(0)
        xVals.append(self.getLength())
        for elem in beam:
            if isinstance(elem, DistributedLoad):
                xVals.append(elem.getPositionPoint1().getX())
                xVals.append(elem.getPositionPoint2().getX())
            else: xVals.append(elem.getPositionPoint().getX())
        return np.unique(xVals).tolist()
    
    def getXVals(self):
        return self.xVals
    
    def makeIntervals(self, xVals):
        """
        Creates a two-dimensional array. The inner arrays represent intervals 
        of a beam. (Here, xVals is a list returned by makeXVals method)
        
        xVals: list of integers.
        Returns: two-dimensional array of intervals of a beam
        """
        intervals = [[xVals[i], xVals[i+1]] for i in range(len(xVals)-1)]
        #Here we have to add another interval with zero length to be able to calculate a force at the end of beam,
        #since isInsideInterval just checks for the force being at the start of interval
        intervals.append([xVals[-1], xVals[-1]])
        return intervals
    
    def getIntervals(self):
        return self.intervals
    
    def makeLoadIntervals(self, intervals):
        """
        Creates a two-dimensional array. The inner arrays represent elements
        acting on each interval. A distributed load can act on two or more different 
        intervals. (Here, intervals is a list returned by makeIntervals method)
        
        intervals: two-dimensional list of intervals.
        Returns: two-dimensional list of elements acting on each interval.
        """
        loadIntervals = []
        for interval in intervals:
            newInterval = []
            for elem in self.beam:
                if elem.isInsideInterval(interval):
                    newInterval.append(elem)
            loadIntervals.append(newInterval)
        return loadIntervals
    
    def getLoadIntervals(self):
        return self.loadIntervals
    
    def makeShearForces(self, loadIntervals):
        """
        Calculates shear force equations of all intervals (shearForces) along 
        with the shear force equations of the intervals before being summed with 
        shear force values of previous intervals (rawSh) and shear force values 
        of previous intervals (vi). 
        
        loadIntervals: A two-dimensional array of elements acting on each interval
        of a beam. (Here, loadIntervals is returned by makeLoadIntervals method)
                        
        Returns: a tuple of shear force equations list, raw shear force equations
        list and shear force constants list.
        """
        shearForces = []
        rawSh = []
        vi = []
        for i in range(len(loadIntervals)):
            poly = np.poly1d([0])
            lowerLimit = self.getIntervals()[i][0]
            for j in range(len(loadIntervals[i])):
                if isinstance(loadIntervals[i][j], DistributedLoad):
                    loadPoly = loadIntervals[i][j].getPolynomial()
                    integral = np.poly1d(np.polyint(loadPoly))
                    lowerLimitVal = integral(lowerLimit)
                    newPoly = np.polyadd(integral, np.poly1d([-lowerLimitVal]))
                    poly = np.polyadd(poly, newPoly)
                else:
                    if not isinstance(loadIntervals[i][j], Moment):
                        loadPoly = loadIntervals[i][j].getPolynomial()
                        poly = np.polyadd(poly, loadPoly)
            rawSh.append(poly)
            #In this part, we add a constant to each equation so that it's attached to the curve of
            #the previous interval. The constant is obtained using the equation of the previous interval
            #evaluated at the end point of the previous interval.
            if i!=0:
                c = shearForces[i-1](self.getIntervals()[i-1][1])
                vi.append(c)
                poly = np.polyadd(poly, c)
            else:
                vi.append(0)
            shearForces.append(poly)
        #Here we remove the extra length-zero interval we created for the last load from
        #self.intervals, self.loadIntervals and exclude the last element of 
        #shearForces created by this zero-length interval.
        self.intervals.pop()
        self.loadIntervals.pop()
        return shearForces[:-1], rawSh[:-1], vi[:-1]
    
    def getShearForces(self):
        return self.shearForces
    
    def setShearForces(self, shearForces):
        self.shearForces = shearForces
        
    def getRawSh(self):
        return self.rawSh
    
    def setRawSh(self, rawSh):
        self.rawSh = rawSh
        
    def getVi(self):
        return self.vi
    
    def setVi(self, vi):
        self.vi = vi
        
    def makeBendingMoments(self):
        """
        Calculates bending moment equations of all intervals (bendingMoments) by
        integrating the shear force equation of each interval along with the 
        bending moment equations of the intervals before being summed with 
        bending moment values of previous intervals (rawBm) and bending moment 
        values of previous intervals (mi).
                        
        Returns: a tuple of bending moment equations list, raw bending moment equations
        list and bending moment constants list.
        """
        bendingMoments = []
        rawBm = []
        mi = [0]
        intervals = self.getIntervals()
        for i in range(len(intervals)):
            #Get the shear force equation of each interval.
            equation = self.getShearForces()[i]
            #integrate the shear force equation to get the raw bending moment equation.
            integral = np.polyint(equation)
            #Evaluate the integral(bending moment equation) in the lower limit of the interval
            #to get the correct part of the bending moment equation.
            lowerLimit = intervals[i][0]
            #(F(x)-F(lowerLimit))
            integral = np.polyadd(integral, np.poly1d([-integral(lowerLimit)]))
            rawBm.append(integral)
            bendingMoments.append(integral)
        for j in range(1, len(intervals)):
            #Here we just have to shift the bending moment equation by adding a constant to it,
            #so that it's attached to the curve piece of the previous interval.
            constant = bendingMoments[j-1](intervals[j-1][1])
            mi.append(constant)
            bendingMoments[j] = np.polyadd(bendingMoments[j], np.poly1d([constant]))
        #Add the user-defined moments along the beam, if there is any, to its interval equation.
        for k in range(len(self.getLoadIntervals())):
            for l in range(len(self.getLoadIntervals()[k])):
                if isinstance(self.getLoadIntervals()[k][l], Moment):
                    bendingMoments[k] = np.polyadd(bendingMoments[k], np.poly1d([self.getLoadIntervals()[k][l].getMoment()]))
        return bendingMoments, rawBm, mi
    
    def getBendingMoments(self):
        return self.bendingMoments
    
    def setBendingMoments(self, bendingMoments):
        self.bendingMoments = bendingMoments
        
    def getRawBm(self):
        return self.rawBm
    
    def setRawBm(self, rawBm):
        self.rawBm = rawBm
        
    def getMi(self):
        return self.mi
    
    def setMi(self, mi):
        self.mi = mi
        
    def calculateReactions(self):
        """
        Calculates the reaction force of a support element along a beam. 
        
        This method is overridden by SimpleBeam and CantileverBeam classes' 
        calculateReactions methods, since they are calculated differently.
        """
        raise NotImplementedError
        

class SimpleBeam(Beam):
    """
    A SimpleBeam is a beam with two supports one being a pinned support and the 
    other a simple or roller support.
    
    This class is a subclass of Beam class and inherits everything from it
    except it overrides its calculateReactions method.
    """
    def calculateReactions(self):
        """
        Calculates the reaction forces of a SimpleBeam supports.
        """
        result = 0
        loads = self.getLoads()
        for load in loads:
            if isinstance(load, PointLoad):
                distanceFromSupport = load.getPositionPoint().distanceFromPoint(self.supports[0].getPositionPoint())
                if self.supports[0].isToTheRight(load):
                    result += -(load.getLoadVectorY().getY() * distanceFromSupport)
                else: result += load.getLoadVectorY().getY() * distanceFromSupport
            if isinstance(load, DistributedLoad):
                distanceFromSupport = load.getResultantLoad().getPositionPoint().\
                distanceFromPoint(self.supports[0].getPositionPoint())
                if self.supports[0].isToTheRight(load.getResultantLoad()):
                    result -= load.getResultantLoad().getLoadVectorY().getY() * distanceFromSupport
                else: result += load.getResultantLoad().getLoadVectorY().getY() * distanceFromSupport
        #Here we have to check if the other support is in the right side of the first one to ensure
        #the sign of the reaction force y-coordinate.
        if self.supports[0].isToTheRight(self.supports[1]):
            self.supports[1].setReactionForce(Vector(0,float(result/self.supports[1].getPositionPoint().\
                        distanceFromPoint(self.supports[0].getPositionPoint())),0))
        else: self.supports[1].setReactionForce(Vector(0,float(-result/self.supports[1].getPositionPoint().\
                    distanceFromPoint(self.supports[0].getPositionPoint())),0))
        
        total = 0
        for load in loads:
            if isinstance(load, PointLoad):
                total += load.getLoadVectorY().vectorMagnitude()
            if isinstance(load, DistributedLoad):
                total += load.getResultantLoad().getLoadVectorY().vectorMagnitude()
        self.supports[0].setReactionForce(Vector(0, float(total - self.supports[1].getReactionForce().getY()), 0))
        
        shearForces, rawSh, vi = self.makeShearForces(self.getLoadIntervals())
        self.setShearForces(shearForces)
        self.setRawSh(rawSh)
        self.setVi(vi)
        
        bendingMoments, rawBm, mi = self.makeBendingMoments()
        self.setBendingMoments(bendingMoments)
        self.setRawBm(rawBm)
        self.setMi(mi)
        

class CantileverBeam(Beam):
    """
    A CantileverBeam is a beam with one fixed support at one of two ends of a
    beam.
    
    This class is a subclass of Beam class and inherits everything from it
    except it overrides its calculateReactions method.
    """
    def calculateReactions(self):
        """
        Calculates the reaction force of a CantileverBeam support.
        """
        loads = self.getLoads()
        total = 0
        
        for load in loads:
            if isinstance(load, PointLoad):
                total += load.getLoadVectorY().getY()
            if isinstance(load, DistributedLoad):
                total += load.getResultantLoad().getLoadVectorY().getY()
        self.supports[0].setReactionForce(Vector(0, float(-total), 0))
        
        shearForces, rawSh, vi = self.makeShearForces(self.getLoadIntervals())
        self.setShearForces(shearForces)
        self.setRawSh(rawSh)
        self.setVi(vi)
        
        bendingMoments, rawBm, mi = self.makeBendingMoments()
        self.setBendingMoments(bendingMoments)
        self.setRawBm(rawBm)
        self.setMi(mi)
        

def polyString(coeffs):
    equation = ''
    if len(coeffs)==1 and coeffs[0]==0:
        equation+='0'
    else:
        for i in range(len(coeffs)):
            if coeffs[i]==0:
                continue
            if coeffs[i]>0 and i!=0:
                equation+='+'
            if coeffs[i]!=1 and coeffs[i]!=-1:
                equation+=str(coeffs[i])
            if coeffs[i]==-1:
                equation+='-'
            if (coeffs[i]==1 or coeffs[i]==-1) and i==len(coeffs)-1:
                equation+='1'
            if i!=len(coeffs)-1 and len(coeffs)-1-i != 1:
                equation+='x^' + str(len(coeffs)-1-i)
            if len(coeffs)-1-i == 1:
                equation+='x'
    return equation


def convertToInt(coeffs):
    decimals = np.modf(coeffs)[0].tolist()
    coeffs = coeffs.tolist()
    for i in range(len(decimals)):
        if decimals[i]==0:
            coeffs[i] = int(coeffs[i])
    return coeffs


def rootFinder(shEq, benEq, intervals):
    roots = []
    rootVals = []
    for i in range(len(shEq)):
        r = np.roots(shEq[i]).tolist()
        if len(r)!=0:
            for root in r:
                if not(type(root)==complex) and root>intervals[i][0] and root<intervals[i][-1]:
                    roots.append(root)
                    rootVals.append(float(np.round(benEq[i](root),2)))
    return [roots, rootVals]


def beamMaker(elemList, beam_type, flPt):
    length, pointLoads, distLoads, supports, moments = 0, [], [], [], []
    for elem in elemList:
        if "Length" in elem.keys():
            length = float(elem["Length"])
            print("This is length of the beam ", length)
        if "PointLoad" in elem.keys():
            pload = PointLoad(Point(float(elem["PointLoad"]["location"]),0,0), float(elem["PointLoad"]["magnitude"]),\
                                float(elem["PointLoad"]["angle"]))
            pointLoads.append(pload)
        if "DistributedLoad" in elem.keys():
            dload = DistributedLoad(Point(float(elem["DistributedLoad"]["locationStart"]),0,0),\
                                    Vector(0,float(elem["DistributedLoad"]["magnitudeStart"]),0),\
                                    Point(float(elem["DistributedLoad"]["locationEnd"]),0,0),\
                                    Vector(0,float(elem["DistributedLoad"]["magnitudeEnd"]),0))
            distLoads.append(dload)
        if "Support" in elem.keys():
            support = Support(Point(float(elem["Support"]["location"]),0,0))
            supports.append(support)
        if "Moment" in elem.keys():
            moment = Moment(Point(float(elem["Moment"]["location"]),0,0), float(elem["Moment"]["magnitude"]))
            moments.append(moment)
            
    beam = beam_type(pointLoads, distLoads, supports, moments, length)
    beam.calculateReactions()
    
    shearXvals, shearYvals, bendingXvals, bendingYvals = [], [], [], []
    shearForces = beam.getShearForces()
    bendingMoments = beam.getBendingMoments()
    
    for i in range(len(shearForces)):
        if shearForces[i].order>1:
            sxvals = np.round(np.linspace(beam.getIntervals()[i][0], beam.getIntervals()[i][1], 100), flPt).tolist()
            syvals = np.round(shearForces[i](sxvals), flPt).tolist()
            shearXvals.append(sxvals)
            shearYvals.append(syvals)
        else:
            sxvals = beam.getIntervals()[i]
            syvals = np.round(shearForces[i](sxvals), flPt).tolist()
            shearXvals.append(sxvals)
            shearYvals.append(syvals)
        if bendingMoments[i].order>1:
            bxvals = np.round(np.linspace(beam.getIntervals()[i][0], beam.getIntervals()[i][1], 100), flPt).tolist()
            byvals = np.round(bendingMoments[i](bxvals), flPt).tolist()
            bendingXvals.append(bxvals)
            bendingYvals.append(byvals)
        else:
            bxvals = beam.getIntervals()[i]
            byvals = np.round(bendingMoments[i](bxvals), flPt).tolist()
            bendingXvals.append(bxvals)
            bendingYvals.append(byvals)

    shearForces, rawSh, bendingMoments, rawBm = [], [], [], []
    for i in range(len(beam.getShearForces())):
        sf = convertToInt(np.round(beam.getShearForces()[i].c, flPt))
        shearEqString = polyString(sf)
        shearForces.append(shearEqString)

        rSf = convertToInt(np.round(beam.getRawSh()[i].c, flPt))
        rawSfString = polyString(rSf)
        rawSh.append(rawSfString)

        bm = convertToInt(np.round(beam.getBendingMoments()[i].c, flPt))
        bendingEqString = polyString(bm)
        bendingMoments.append(bendingEqString)

        rBm = convertToInt(np.round(beam.getRawBm()[i].c, flPt))
        rawBmString = polyString(rBm)
        rawBm.append(rawBmString)

    roots = rootFinder(beam.getShearForces(), beam.getBendingMoments(), beam.getIntervals())[0]
    rootVals = rootFinder(beam.getShearForces(), beam.getBendingMoments(), beam.getIntervals())[1]
    xVals = beam.getXVals()
    xVals.extend(roots)
    sy = [s[0] for s in shearYvals]
#    sy.append(shearYvals[-1][-1])
    by = [b[0] for b in bendingYvals]
#    by.append(bendingYvals[-1][-1])
    by.extend(rootVals)
    #Flattening the x and y values of shear force and bending moment points.
    shearXvals = [x for xvals in shearXvals for x in xvals]
    shearYvals = [y for yvals in shearYvals for y in yvals]
    bendingXvals = [x for xvals in bendingXvals for x in xvals]
    bendingYvals = [y for yvals in bendingYvals for y in yvals]
    return [shearForces, bendingMoments, shearXvals, shearYvals, bendingXvals, bendingYvals,\
            xVals, sy, by, rawSh, np.round(beam.getVi(), flPt).tolist(), rawBm, \
            np.round(beam.getMi(), flPt).tolist()]

#*** @staticmethod won't work when using Python as a child process in Node js; we need to use
#the usual type of function for this purpose.***
#***Main test
#if __name__=="__main__":
#    input = sys.stdin.readlines()
#    input = json.loads(input[0])
#    print(json.dumps(beamMaker(input)))
#    sys.stdout.flush()

# dataFromClient = [{"Length": 20}, {"PointLoad": {"location": 0, "magnitude": -8, "angle": 90}},\
#                     {"DistributedLoad": {"locationStart": 4, "locationEnd": 20, "magnitudeStart": -3, "magnitudeEnd": -3}},\
#                     {"Support": {"location": 4}}, {"Support": {"location": 20}}]

# dataFromClient = [{"Length": 3}, {"PointLoad": {"location": 0, "magnitude": -33, "angle": 90}},\
#                     {"Support": {"location": 0}}, {"Support": {"location": 3}}]
#
#dataFromClient = [{"Length": 3}, {"DistributedLoad": {"locationStart": 0, "locationEnd": 3, "magnitudeStart": -3, "magnitudeEnd": -3}},\
#                  {"Support": {"location": 0.5}}, {"Support": {"location": 2.5}}]

dataFromClient = [{"Length": 4}, {"DistributedLoad": {"locationStart": 0, "locationEnd": 4, "magnitudeStart": -3, "magnitudeEnd": -3}},\
                  {"PointLoad": {"location": 2, "magnitude": 15, "angle": 90}}, {"Support": {"location": 0}}]

# dataFromClient = [{"Length": 3}, {"PointLoad": {"location": 2, "magnitude": -33, "angle": 90}},\
#                     {"PointLoad": {"location": 1, "magnitude": -22, "angle": 90}},\
#                     {"Support": {"location": 0}}, {"Support": {"location": 3}}]

#dataFromClient = [{"Length": 3}, {"DistributedLoad": {"locationStart": 1, "locationEnd": 3, "magnitudeStart": -33, "magnitudeEnd": 0}},{"Support": {"location": 0}}, {"Support": {"location": 3}}]

#beam = beamMaker(dataFromClient, SimpleBeam, 2)
beam = beamMaker(dataFromClient, CantileverBeam, 2)
print(beam)
