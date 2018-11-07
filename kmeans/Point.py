class Point:

    def __init__(self, coords):
        self.coords = coords

    def distance_to(self, other_point):
        for i in other_point.coords:
            print self.coords
            print i
