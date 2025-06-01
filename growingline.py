
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

        self.switch = []
        self.in_switch = False

        self.points = [head_xyz]

        self.id = GrowingLine._id_counter
        GrowingLine._id_counter += 1

    @classmethod
    def from_feature(cls, feature):
        head_fid = feature.GetFID()
        head_xyz = np.array(feature.GetGeometryRef().GetPoint(0))
        head_direction = np.array([feature.GetField("eig x"), feature.GetField("eig y"), feature.GetField("eig z")])
        return cls(head_fid, head_xyz, head_direction)



    def points_in_direction(self, layer, distance=10):
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

        return xyz, directions, fids


    def add_switch(self, new_fid, head_xyz,  new_direction, first_point):


        if self.in_switch:
            # Update the switch line
            switchline = self.switch[-1] 
            switchline.head_fid = new_fid
            switchline.head_xyz = head_xyz
            switchline.head_direction = new_direction
        else:
            print("Adding new switch line")
            self.in_switch = True
            switchline = GrowingLine(
                new_fid, head_xyz, new_direction)
            switchline.points = [first_point]
            self.switch.append(switchline)


    def reset(self):
        self.head_fid = self.start_fid
        self.head_xyz = self.start_xyz
        self.head_direction = self.start_direction 
        self.points = [self.start_xyz]

    def make_cut(self, first_fid, first_xyz, first_direction):
        print("Making cut in switch")
        self.in_switch = False
        switchline = self.switch[-1]

        length = np.linalg.norm(switchline.head_xyz - switchline.points[0])
        print("Length of switch line:", length, "points:", len(switchline.points))

        between_heads = self.head_xyz - switchline.head_xyz
        # Check if the active line ends in the switch 
        # it happens if we approach the switch from the curved track
        cut_active_line = (between_heads @ self.head_direction) > -1.5
        print("Cut active line:", cut_active_line)
        distance_from_head = np.linalg.norm(between_heads)

        if cut_active_line:
            print("Distance from head to switch line head:", distance_from_head)
            
            go_back = distance_from_head + length + 8
            print("Going back:", go_back)
            reversed_points = self.points[::-1]

            distances = []
            cum_sum = 0
            
            for i in range(len(reversed_points)-1):
                distance = np.linalg.norm(reversed_points[i] - reversed_points[i+1])
                cum_sum += distance
                distances.append(distance)
                if cum_sum > go_back:
                    break

            distances = np.array(distances)
            cut_index = np.argmax(distances) 
            print("Cut index:", cut_index, "distance", distances[cut_index], "go_back", go_back)

            # Reset the switchline
            switchline.reset()
            if switchline.head_direction @ self.head_direction < 0:
                switchline.head_direction = -switchline.head_direction

            # Cut the line

            self.points = reversed_points[:cut_index:-1]
            new_points = reversed_points[cut_index::-1]

            new_line = GrowingLine(
                self.head_fid,
                self.head_xyz,
                self.head_direction,
            )
            new_line.points = new_points
            # New line doesn't need to be reversed
            new_line.start_fid = None
            self.switch.append(new_line)

        else:
            # Set the switchline to the given point (first point of cluster)
            # And make shure it will not reverse
            if first_direction @ switchline.head_direction < 0:
                first_direction = -first_direction
            switchline.head_fid = first_fid
            switchline.head_xyz = first_xyz
            switchline.head_direction = first_direction
            switchline.points = [first_xyz]
            switchline.start_fid = None



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
        
        print("Reversed head")
        return True


    def get_linestring(self):
        """Get the linestring of the line
        
        """
        if len(self.points) == 0:
            return None
        geom = ogr.Geometry(ogr.wkbLineString25D)
        for point in self.points:
            geom.AddPoint(point[0], point[1], point[2])
        return geom

    def grow(self, layer, linelayer):
        first_fid = None
        first_xyz = None 
        first_direction = None
        first_iteration = True

        while True:
            xyz, directions, fids = self.points_in_direction(layer)
            if len(fids) < 3:
                # These are only 2 new points, not enough for a ransac line
                # Reverse head or stop if already reversed
                remove_points(fids, layer)
                if self.in_switch:
                    self.make_cut(first_fid, first_xyz, first_direction)
                if not self.reverse_head():
                    # If we can't reverse the head, we are done
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
            if self.in_switch and max_label == 0:
                # Use the first point of the last round if cut is on the switchline
                self.make_cut(first_fid, first_xyz, first_direction)
                if not self.reverse_head():
                    # If we can't reverse the head, we are done
                    break
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
                        self.add_switch(fids_cluster[-1], cluster[-1], directions_cluster[-1], cluster[0])

                        # Keep first point for the cut method
                        first_fid = fids_cluster[0]
                        first_xyz = cluster[0]
                        first_direction = directions_cluster[0]

        # Add the active line to the layer if it has enough points
        geom = self.get_linestring()
        if geom is not None and geom.GetPointCount() > 10:
            linelayer_add(linelayer, geom)

        return self.switch
    
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

def ransac_lines(points, threshold=0.1, min_inliers=3, max_iterations=20, max_lines=2):
    points = np.asarray(points)
    N = len(points)
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

def linelayer_add(layer, geom):
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)
