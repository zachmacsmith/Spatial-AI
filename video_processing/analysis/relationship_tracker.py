"""
Relationship Tracker - Track object proximity relationships over time

Tracks when objects are close to each other (within proximity threshold)
and records the duration of these relationships.

Extracted from TestingClass3.py
"""

import numpy as np
from typing import List, Tuple, Dict, Set, FrozenSet


class RelationshipTracker:
    """
    Tracks relationships between objects based on proximity.
    
    A relationship exists when objects are within proximity_threshold_percent
    of the frame width from each other.
    """
    
    def __init__(self, fps: float, proximity_threshold_percent: float):
        """
        Initialize relationship tracker.
        
        Args:
            fps: Video frames per second
            proximity_threshold_percent: Proximity threshold as % of frame width (e.g., 0.18 = 18%)
        """
        self.fps = fps
        self.proximity_threshold_percent = proximity_threshold_percent
        
        # Active relationships: key = frozenset of object names, value = start_frame
        self.active_relationships: Dict[FrozenSet[str], int] = {}
        
        # Completed relationships: list of {objects, start_frame, end_frame}
        self.completed_relationships: List[Dict] = []
    
    def get_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centers"""
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    
    def find_relationships(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], str, float]],
        frame_width: int,
        current_frame: int
    ) -> Tuple[List[FrozenSet[str]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """
        Find relationships between objects in current frame.
        
        Args:
            detections: List of (box, class_name, confidence) where box is (x1, y1, x2, y2)
            frame_width: Width of frame in pixels
            current_frame: Current frame number
        
        Returns:
            Tuple of (relationship_sets, line_info) where:
            - relationship_sets: List of frozensets of object names that are related
            - line_info: List of (center1, center2) tuples for drawing lines
        """
        proximity_threshold = frame_width * self.proximity_threshold_percent
        
        if len(detections) < 2:
            return [], []
        
        # Calculate centers for all detections
        centers = [(self.get_box_center(det[0]), det[1]) for det in detections]
        
        # Build adjacency: which objects are close to each other
        n = len(centers)
        adjacency = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_distance(centers[i][0], centers[j][0])
                if dist <= proximity_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # Find connected components (groups of close objects) using DFS
        visited = [False] * n
        relationships = []
        
        def dfs(node: int, component: Set[int]):
            visited[node] = True
            component.add(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n):
            if not visited[i] and adjacency[i]:  # Has at least one close neighbor
                component: Set[int] = set()
                dfs(i, component)
                if len(component) >= 2:
                    relationships.append(component)
        
        # Convert indices to object names and prepare line drawing info
        relationship_sets = []
        line_info = []
        
        for component in relationships:
            obj_names = frozenset(centers[idx][1] for idx in component)
            relationship_sets.append(obj_names)
            
            # Prepare lines to draw between all pairs in component
            indices = list(component)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    center1 = centers[indices[i]][0]
                    center2 = centers[indices[j]][0]
                    line_info.append((center1, center2))
        
        return relationship_sets, line_info
    
    def update(self, relationships: List[FrozenSet[str]], current_frame: int):
        """
        Update relationship tracking with current frame's relationships.
        
        Args:
            relationships: List of frozensets of object names that are currently related
            current_frame: Current frame number
        """
        current_relationship_keys = set(relationships)
        
        # Start new relationships or continue existing ones
        for rel in relationships:
            if rel not in self.active_relationships:
                # New relationship detected - start tracking it
                self.active_relationships[rel] = current_frame
                print(f"Frame {current_frame}: Started tracking relationship: {set(rel)}")
        
        # Check for ended relationships
        ended_keys = []
        for rel_key, start_frame in self.active_relationships.items():
            if rel_key not in current_relationship_keys:
                # Relationship ended - record it
                self.completed_relationships.append({
                    'objects': rel_key,
                    'start_frame': start_frame,
                    'end_frame': current_frame - 1  # Last frame where it was active
                })
                ended_keys.append(rel_key)
                print(f"Frame {current_frame}: Ended relationship: {set(rel_key)} "
                      f"(lasted {current_frame - start_frame} frames)")
        
        for key in ended_keys:
            del self.active_relationships[key]
    
    def finalize(self, last_frame: int):
        """
        Finalize all active relationships at end of video.
        
        Args:
            last_frame: Last frame number in video
        """
        for rel_key, start_frame in self.active_relationships.items():
            self.completed_relationships.append({
                'objects': rel_key,
                'start_frame': start_frame,
                'end_frame': last_frame
            })
            print(f"Finalized relationship: {set(rel_key)} "
                  f"(lasted {last_frame - start_frame + 1} frames)")
        self.active_relationships.clear()
    
    def get_relationships_csv_data(self) -> List[Dict]:
        """
        Get relationship data formatted for CSV output.
        
        Returns:
            List of dicts with keys: objects, start_frame, end_frame
        """
        return self.completed_relationships
