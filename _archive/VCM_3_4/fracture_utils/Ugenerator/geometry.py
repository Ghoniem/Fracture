"""
Geometry utilities for crack network operations

This module contains geometric algorithms for:
- Line segment intersection detection
- Distance calculations between segments
- Angle computations
- Coordinate transformations
"""

import numpy as np
from typing import Tuple, Optional
from fracture_utils.Usolver.network import CrackNetworkV4 as CrackNetworkV4


class GeometryUtils:
    """Utility class for geometric operations on crack networks"""
    
    @staticmethod
    def segments_intersect(p1: np.ndarray, p2: np.ndarray, 
                          p3: np.ndarray, p4: np.ndarray,
                          tolerance: float = 0.05) -> Optional[np.ndarray]:
        """
        Find intersection point of two line segments
        
        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment
            tolerance: Minimum distance from endpoints (as fraction, default 0.05 = 5%)
            
        Returns:
            Intersection point [x, y] or None if no intersection
        """
        # Line segment intersection algorithm
        # Segment 1: p1 + t * (p2 - p1), t in [0, 1]
        # Segment 2: p3 + s * (p4 - p3), s in [0, 1]
        
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p1 - p3
        
        # Cross product for 2D
        cross_d1_d2 = d1[0] * d2[1] - d1[1] * d2[0]
        
        # Parallel or coincident lines
        if abs(cross_d1_d2) < 1e-10:
            return None
        
        # Calculate parameters
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross_d1_d2
        s = (d3[0] * d1[1] - d3[1] * d1[0]) / cross_d1_d2
        
        # Check if intersection is within both segments (excluding endpoints)
        if tolerance < t < 1.0 - tolerance and tolerance < s < 1.0 - tolerance:
            # Calculate intersection point
            intersection = p1 + t * d1
            return intersection
        
        return None
    
    @staticmethod
    def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """Check if three points are counterclockwise"""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    @staticmethod
    def segments_intersect_bool(A: np.ndarray, B: np.ndarray, 
                                C: np.ndarray, D: np.ndarray) -> bool:
        """
        Check if line segments AB and CD properly intersect (boolean)
        
        Returns True if they intersect at interior points
        """
        ccw = GeometryUtils.ccw
        
        # Check basic intersection
        if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
            # They intersect - check if at endpoints (allowed)
            tol = 1e-10
            for P1 in [A, B]:
                for P2 in [C, D]:
                    dist = np.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)
                    if dist < tol:
                        return False  # Touching at endpoint is OK
            return True  # True interior intersection
        return False
    
    @staticmethod
    def segment_to_segment_distance(p1: np.ndarray, p2: np.ndarray,
                                    p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Calculate minimum distance between two line segments
        
        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment
            
        Returns:
            Minimum distance between the segments
        """
        # Vector representations
        d1 = p2 - p1  # Direction of segment 1
        d2 = p4 - p3  # Direction of segment 2
        r = p1 - p3
        
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        # Check if either or both segments degenerate into points
        epsilon = 1e-10
        
        if a <= epsilon and e <= epsilon:
            # Both segments are points
            return np.linalg.norm(p1 - p3)
        
        if a <= epsilon:
            # First segment is a point
            s = 0.0
            t = np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            if e <= epsilon:
                # Second segment is a point
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            else:
                # General case
                b = np.dot(d1, d2)
                denom = a * e - b * b
                
                # If segments not parallel, compute closest point on infinite lines
                if denom != 0.0:
                    s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    # Parallel segments - pick arbitrary point on first segment
                    s = 0.0
                
                # Compute point on second segment closest to s
                t = (b * s + f) / e
                
                # If t outside [0,1], clamp and recompute s
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - c) / a, 0.0, 1.0)
        
        # Compute the closest points
        closest_point_1 = p1 + s * d1
        closest_point_2 = p3 + t * d2
        
        return np.linalg.norm(closest_point_1 - closest_point_2)
    
    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate angle between two vectors in degrees.

        Returns NaN when either input vector has zero (or sub-eps) length;
        the angle is undefined for coincident points.

        Args:
            v1, v2: Two 2D vectors

        Returns:
            Angle in degrees in [0, 180], or NaN for degenerate input.
        """
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-30 or n2 < 1e-30:
            return float("nan")

        cos_angle = np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.rad2deg(angle_rad))
    
    @staticmethod
    def minimum_angle_at_junction(junction_pos: np.ndarray, 
                                  neighbor_positions: list) -> float:
        """
        Calculate minimum angle between edges at a junction
        
        Args:
            junction_pos: Position of junction vertex
            neighbor_positions: List of neighbor vertex positions
            
        Returns:
            Minimum angle in degrees
        """
        if len(neighbor_positions) < 2:
            return 180.0  # No angle constraint for single edge
        
        min_angle = 180.0
        
        for i, pos1 in enumerate(neighbor_positions):
            vec1 = pos1 - junction_pos
            vec1_norm = vec1 / np.linalg.norm(vec1)
            
            for pos2 in neighbor_positions[i+1:]:
                vec2 = pos2 - junction_pos
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                angle = GeometryUtils.angle_between_vectors(vec1_norm, vec2_norm)
                # Check both angle and its supplement
                min_separation = min(angle, 180 - angle)
                min_angle = min(min_angle, min_separation)
        
        return min_angle

# -------------------------
# POLYLINE arc geometry (nseg segments)
# returns (net, V) so we can grab p_tip robustly
# -------------------------
def polyline_arc_network(alpha_half, a_chord, nseg=12):
    R = a_chord / np.sin(alpha_half)
    yc = R * np.cos(alpha_half)

    thL = np.pi/2 + alpha_half
    thR = np.pi/2 - alpha_half
    th  = np.linspace(thL, thR, nseg + 1)

    x = R * np.cos(th)
    y = -yc + R * np.sin(th)

    V = np.zeros((nseg + 1, 3), float)
    V[:,0] = np.arange(nseg + 1)   # ids
    V[:,1] = x
    V[:,2] = y

    Econn = np.array([[i, i+1] for i in range(nseg)], int)

    net = CrackNetworkV4.from_vertices_connectivity(
        vertices=V,
        connectivity=Econn,
        Nv_max=nseg + 1,
        validate=True,
    )
    return net, V