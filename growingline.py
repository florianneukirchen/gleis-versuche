
import sys
import os
import numpy as np
from osgeo import ogr 
from collections import deque
import matplotlib.pyplot as plt


sys.path.append("/home/riannek/code/gleis/gleisachse")
from algs.gpkg import * 

ogr.UseExceptions()


class GrowingLine: 

    _id_counter = 0 # Used for debugging

    def __init__(self, head_fid, head_xyz, head_direction):
        self.head_fid = head_fid
        self.head_xyz = head_xyz
        self.head_direction = head_direction

        self.start_xyz = head_xyz
        self.start_direction = head_direction
        self.start_fid = head_fid

        self.switch = None
        self.next_lines = []

        self.points = [head_xyz]

        self.id = GrowingLine._id_counter
        GrowingLine._id_counter += 1

    @classmethod
    def from_feature(cls, feature):
        head_fid = feature.GetFID()
        head_xyz = np.array(feature.GetGeometryRef().GetPoint(0))
        head_direction = np.array([feature.GetField("eig x"), feature.GetField("eig y"), feature.GetField("eig z")])
        return cls(head_fid, head_xyz, head_direction)

    def in_switch(self):
        return self.switch is not None 

    def points_in_direction(self, layer, switchlayer, distance=10):
        """Read the points in the model direction and sort them
        
        """

        # Get a rectangular polygon for the spatial filter
        perpendicular = np.cross(self.head_direction, np.array([0, 0, 1]))
        scaled_direction = self.head_direction * distance

        ring = ogr.Geometry(ogr.wkbLinearRing)
        start = self.head_xyz + (perpendicular / 2)
        ring.AddPoint(start[0], start[1])
        pt = start + scaled_direction
        ring.AddPoint(pt[0], pt[1])
        pt = pt - perpendicular
        ring.AddPoint(pt[0], pt[1])
        pt = pt - scaled_direction
        ring.AddPoint(pt[0], pt[1])
        ring.AddPoint(start[0], start[1])
        geom = ogr.Geometry(ogr.wkbPolygon)
        geom.AddGeometry(ring)

        layer.SetSpatialFilter(geom)
        layer.ResetReading()

        xyz = [self.head_xyz]
        directions = [self.head_direction]
        fids = [self.head_fid]
        distances = [0]

        for feature in layer:
            if feature.GetFID() == self.head_fid:
                continue
            xyz.append(np.array(feature.GetGeometryRef().GetPoint(0)))
            directions.append(np.array([feature.GetField("eig x"), feature.GetField("eig y"), feature.GetField("eig z")]))
            fids.append(feature.GetFID())

        layer.SetSpatialFilter(None)
        layer.ResetReading()


        for point in xyz[1:]:
            distances.append(np.linalg.norm(self.head_xyz - point))

        # Sort all lists by distance
        distances = np.array(distances)
        sorted_indices = np.argsort(distances)
        distances = distances[sorted_indices]
        xyz = np.array(xyz)[sorted_indices]
        directions = np.array(directions)[sorted_indices]
        fids = np.array(fids)[sorted_indices]

        # Read points from switchlayer
        switches = []
        switchlayer.ResetReading()
        switchlayer.SetSpatialFilter(geom)
        for feature in switchlayer:
            switches.append(np.array(feature.GetGeometryRef().GetPoint(0)))

        return xyz, directions, fids, switches


    def add_switchline(self, head_xyz, first_point):

        if not self.switch is None:
            # Update the switch line
            self.switch.append(head_xyz)
        else:
            self.switch = [first_point, head_xyz]

    def index_closest_point(self, switchpoints):

        active_line = np.vstack(self.points)
    
        indices = []
        distances = []

        for point in switchpoints:
            print("POINT")
            print(point)
            dist = np.linalg.norm(active_line - point, axis=1)
            idx = np.argmin(dist)
            distances.append(dist[idx])
            indices.append(idx)

        # Index with closest distance of any switchpoint
        distances = np.array(distances)
        return indices[np.argmin(distances)] 


    def make_cut(self, switchpoints=None):


        if switchpoints is None: 
            switchpoints = self.switch

        cut_index = self.index_closest_point(switchpoints)
    
        # Cut the line
        cut_point = self.points[cut_index]
        new_points = self.points[cut_index:]
        self.points = self.points[:cut_index]

        new_line = GrowingLine(
            self.head_fid,
            self.head_xyz,
            self.head_direction,
        )
        new_line.points = new_points
        # New line doesn't need to be reversed
        new_line.start_fid = None

        self.next_lines.append(new_line)
        self.switch = None
        return cut_point


    def reverse_head(self, active_line=True):
        """Reverse the head of the line
        
        """
        
        if self.start_fid is None:
            print("FINISHED")
            return False
        self.head_fid = self.start_fid
        self.head_xyz = self.start_xyz
        self.head_direction = -self.start_direction
        self.points = self.points[::-1]
        
        if active_line:
            self.start_fid = None
        
        print("Reversed head", self.head_fid)
        return True



    def grow(self, layer, linelayer, switchlayer):

        first_iteration = True

        while True:
            xyz, directions, fids, switches = self.points_in_direction(layer, switchlayer)
            if len(fids) < 3:
                # These are only 2 new points, not enough for a ransac line
                remove_points(fids, layer)
                if self.in_switch():
                    # Write the head of the line to the switchlayer
                    geom = ogr.Geometry(ogr.wkbPoint)
                    geom.AddPoint(self.head_xyz[0], self.head_xyz[1])
                    add_to_layer(switchlayer, geom)

                # Reverse head or break out of loop if already reversed
                if not self.reverse_head():
                    break
                continue

            labels = ransac_lines(xyz, threshold=0.05, max_iterations=20)
            max_label = labels.max()

            # If the start point is in a switch, it causes all kinds of problems 
            if first_iteration and max_label > 0:
                raise StartInSwitchError
            first_iteration = False

            remove_points(fids[labels == -1], layer)

            # Check if we reached the end of a switch
            if self.in_switch() and max_label == 0:
                if labels[0] == -1:
                    # The active line ended in the switch 
                    # Check which end of the switchline is closer and write the 
                    # closest point to the switch layer
                    #idx = self.index_closest_point([self.switch[0], self.switch[1]])
                    # closest_point = self.points[idx]
                    geom = ogr.Geometry(ogr.wkbPoint)
                    geom.AddPoint(self.head_xyz[0], self.head_xyz[1])
                    # geom.AddPoint(closest_point[0], closest_point[1])
                    add_to_layer(switchlayer, geom)
                    # Reverse head or break out of loop if already reversed
                    if not self.reverse_head():
                        break
                else:
                    # The switchline ended
                    # If it was very short, it is a false positive
                    length = np.linalg.norm(self.switch[0] - self.switch[-1])

                    if length > 5:
                        point = self.make_cut()
                        geom = ogr.Geometry(ogr.wkbPoint)
                        geom.AddPoint(point[0], point[1])
                        add_to_layer(switchlayer, geom)

                        # Reverse head or break out of loop if already reversed
                        if not self.reverse_head():
                            break
                    else:
                        # The switch line is too short, remove it
                        print("Removing switch line, too short:", length)
                        self.switch = None

            # Add the points to the line(s)
            else:
                for label in range(max_label + 1):
                    cluster = xyz[labels == label]
                    if len(cluster) < 2:
                        continue

                    fids_cluster = fids[labels == label]
                    directions_cluster = directions[labels == label]

                    if label == labels[0]:
                        # This is the active head
                        pruned, offset = pruned_points(cluster) 

                        remove_points(fids_cluster[:offset+1], layer)
                    
                        new_direction = directions_cluster[offset]
                        if self.head_direction @ new_direction < 0:
                            new_direction = -new_direction

                        self.head_xyz = pruned[-1]
                        self.head_direction = new_direction
                        self.head_fid = fids_cluster[offset]
                        if np.array_equal(pruned[0], self.points[-1]):
                            # Remove the first point if it is already in the line
                            pruned = pruned[1:]  
                        self.points.extend(pruned)
                    else:
                        # This is the other rail in a switch (or false positive)
                        self.add_switchline(cluster[-1], cluster[0])



                # Make a cut if there is a switchpoint
                if len(switches) > 0:
                    self.make_cut(switches)
                    # Reverse head or break out of loop if already reversed
                    if not self.reverse_head():
                        break

        # Finished Growing

        # Add the active line to the layer if it has enough points
        geom = self.get_linestring()
        if geom is not None and geom.GetPointCount() > 10:
            add_to_layer(linelayer, geom)

        return self.next_lines


    def get_linestring(self):
        """Get the linestring of the line
        
        """
        if len(self.points) == 0:
            return None
        geom = ogr.Geometry(ogr.wkbLineString25D)
        for point in self.points:
            geom.AddPoint(point[0], point[1], point[2])
        return geom


    def __repr__(self):
        return f"GrowingLine(id={self.id}, head_fid={self.head_fid}, points={len(self.points)})"


class StartInSwitchError(Exception):
    """Exception raised if start point is in a switch"""

    def __init__(self, message="Start point is in a switch, prone to bugs."):
        self.message = message
        super().__init__(self.message)    





def model_fitness(xyz, directions, labels=None):
    if labels is None:
        labels = np.zeros(len(xyz), dtype=int)
    fitness = np.zeros(len(xyz))

    for label in range(labels.max()+1):
        cluster = xyz[labels == label]
        # Set first points fitness to 1
        cluster_fitness = [1]

        for i in range(len(cluster)-1):
            cluster_fitness.append(model_fitness_score(cluster[i], cluster[i+1], directions[i]))

        cluster_fitness = np.array(cluster_fitness)
        fitness[labels == label] = cluster_fitness

    if len(fitness[fitness>0]) > 0:
        print("Min fitness:", fitness[fitness>0].min())
    return fitness


def model_fitness_score(point1, point2, direction):
    vector = point1 - point2
    vector = vector / np.linalg.norm(vector)
    direction = direction / np.linalg.norm(direction)
    return np.abs(direction @ vector)    


def distance_points_to_line(points, p0, direction):
    # Vectors from p0 to points
    v = points - p0

    # Line-to-point distance is norm of cross product over norm of direction
    cross = np.cross(v, direction)
    distances = np.linalg.norm(cross, axis=1) / np.linalg.norm(direction)
    return distances

def ransac_lines(points, threshold=0.1, min_inliers=None, max_iterations=20, max_lines=2):
    points = np.asarray(points)
    N = len(points)

    if min_inliers is None:
        if N > 15: 
            min_inliers = 6
        else:
            min_inliers = 3

    labels = np.full(N, -1)  # Initialize all as noise
    remaining_idx = np.arange(N)


    for current_label in range(max_lines):
        best_inliers = []

        for _ in range(max_iterations):
            if len(remaining_idx) < min_inliers:
                break
            sample_idx = np.random.choice(remaining_idx, 2, replace=False)
            p0, p1 = points[sample_idx]
            direction = p1 - p0

            # Direction not accurate if points are too close
            if np.linalg.norm(direction) < 0.1:
                continue

            distances = distance_points_to_line(points[remaining_idx], p0, direction)
            inliers = np.where(distances < threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        if len(best_inliers) < min_inliers:
            break  # No more good lines

        inlier_global_idx = remaining_idx[best_inliers]
        labels[inlier_global_idx] = current_label

        # Remove inliers from remaining set
        remaining_idx = np.setdiff1d(remaining_idx, inlier_global_idx)

        if len(remaining_idx) < min_inliers:
            break

    return labels



def distance_points(point1, point2):
    return np.linalg.norm(point1 - point2)


def pruned_points(xyz):
    pruned_points = [] 
    pruned_points.append(xyz[0]) 
    offset = 0 # Keep reference of the index in the original xyz array
    N = len(xyz)

    while True:
        if len(xyz) < 2:
            # We have only one point left
            return pruned_points, offset

        distances = []
        for i in range(len(xyz)-1):
            distances.append(distance_points(xyz[i], xyz[i+1]))

        distances = np.array(distances)
        distances_cumsum = distances.cumsum()


        if distances_cumsum[0] > 1:
            # The second point is already far away from the first
            pruned_points.append(xyz[1])
            xyz = xyz[1:]
            offset += 1
            continue

        if distances_cumsum[-1] <= 1:
            # The last point is too close
            if offset == 0:
                # We seem to be at the start or end of the line
                # Add the last point
                pruned_points.append(xyz[-1]) 
                offset = N - 1
            return pruned_points, offset

        # Find the index of the last point that is less than 1 m away
        index = np.where(distances_cumsum < 1)[0][-1] + 1
        pruned_points.append(xyz[index])
        xyz = xyz[index:]
        offset += index


def remove_points(fids, layer):
    """Remove points from the layer
    
    """
    if len(fids) == 0:
        return
    first = layer.GetFeature(fids[0])
    if first is None:
        # After reversing, the first point has already been removed
        fids = fids[1:]

    for fid in fids:
        layer.DeleteFeature(fid)

    layer.SyncToDisk()

def add_to_layer(layer, geom):
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)
