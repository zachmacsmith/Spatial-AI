"""
Action Mapper
Maps detected objects and relationships to likely actions.
"""
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class ActionRule:
    required_objects: Set[str]
    action: str
    description: str
    confidence: float = 1.0  # 1.0 = definitive, < 1.0 = suggestion

# Initial Database
ACTION_DATABASE: List[ActionRule] = [
    ActionRule(
        required_objects={"pencil", "tape"},
        action="measuring",
        description="User has both a pencil and tape measure, likely measuring and marking.",
        confidence=0.9
    ),
    ActionRule(
        required_objects={"pencil", "measuring tape"},
        action="measuring",
        description="User has both a pencil and tape measure, likely measuring and marking.",
        confidence=0.9
    ),
    ActionRule(
        required_objects={"drill", "screw"},
        action="fastening",
        description="User has a drill and screws, likely fastening.",
        confidence=0.8
    ),
    ActionRule(
        required_objects={"brick trowel", "brick"},
        action="laying brick",
        description="User has a trowel and brick, likely laying bricks.",
        confidence=0.9
    ),
    ActionRule(
        required_objects={"caulk gun"},
        action="sealing",
        description="User has a caulk gun, likely sealing or caulking.",
        confidence=0.7
    ),
    ActionRule(
        required_objects={"saw"},
        action="cutting",
        description="User has a saw, likely cutting material.",
        confidence=0.7
    ),
    ActionRule(
        required_objects={"nail gun"},
        action="fastening",
        description="User has a nail gun, likely fastening.",
        confidence=0.9
    )
]

class ActionMapper:
    def __init__(self):
        self.rules = ACTION_DATABASE

    def get_definitive_action(self, detected_objects: List[str]) -> Optional[str]:
        """
        Return an action if a high-confidence rule is met.
        detected_objects: list of class names (e.g. ['pencil', 'tape'])
        """
        detected_set = set(obj.lower() for obj in detected_objects)
        
        # Look for exact matches of required objects
        # We prioritize rules with more specific requirements (more objects)
        sorted_rules = sorted(self.rules, key=lambda r: len(r.required_objects), reverse=True)
        
        for rule in sorted_rules:
            if rule.confidence >= 0.9:
                if rule.required_objects.issubset(detected_set):
                    return rule.action
        return None

    def get_likely_actions(self, detected_objects: List[str]) -> List[str]:
        """
        Return a list of likely actions based on detected objects.
        """
        detected_set = set(obj.lower() for obj in detected_objects)
        likely_actions = set()
        
        for rule in self.rules:
            # If any of the required objects are present, it's a candidate
            # But for multi-object rules, we might want stricter logic.
            # For now, if ALL required objects are present, add it.
            if rule.required_objects.issubset(detected_set):
                likely_actions.add(rule.action)
                
        return list(likely_actions)
