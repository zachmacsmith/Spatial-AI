"""
Example: Adding Custom Prompting Architecture

This example shows how to add a new prompting architecture (ReAct)
and compare it with existing methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processing import BatchParameters, process_video
from video_processing.ai.prompt_builder import PromptBuilder
from typing import List, Tuple, Optional
import numpy as np

# ============================================================================
# Step 1: Extend PromptBuilder with New Architecture
# ============================================================================

class ReActPromptBuilder(PromptBuilder):
    """
    ReAct (Reasoning + Acting) Prompting Architecture
    
    Encourages the model to:
    1. Think about observations
    2. Reason about what they mean
    3. Take action (classify)
    """
    
    def __init__(self, batch_params):
        super().__init__(batch_params)
        # Add ReAct to available templates
        self.prompt_templates['react'] = self._build_react_prompt
    
    def _build_react_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Build ReAct-style prompt"""
        
        if task == 'action_classification':
            actions = ", ".join(self.batch_params.allowed_actions)
            
            # Build observations
            observations = []
            
            if motion_score is not None:
                if motion_score < 0.16:
                    motion_desc = "very low (person likely idle)"
                elif motion_score < 0.5:
                    motion_desc = "moderate (person moving slowly)"
                else:
                    motion_desc = "high (person actively moving)"
                observations.append(f"Motion: {motion_score:.2f} - {motion_desc}")
            
            if detected_objects:
                obj_list = ", ".join([obj[0] for obj in detected_objects[:5]])
                observations.append(f"Objects detected: {obj_list}")
            
            # Build ReAct prompt
            prompt = f"""Analyze this construction site video using step-by-step reasoning:

Thought 1: What do I observe in these frames?
Observation: {'; '.join(observations) if observations else 'Analyzing visual content'}

Thought 2: What patterns indicate the worker's activity?
Observation: [Consider motion patterns, tool usage, body posture]

Thought 3: Which category best matches these observations?
Available categories: {actions}

Action: [State ONLY the category name, nothing else]
"""
            return prompt
        
        elif task == 'tool_detection':
            tools = ", ".join(self.batch_params.allowed_tools)
            
            prompt = f"""Identify the tool being used:

Thought: What tools are visible in the frames?
Observation: [Examine hands, nearby objects, work context]

Thought: Which tool from the list matches what I see?
Available tools: {tools}

Action: [State ONLY the tool name]
"""
            return prompt
        
        return super()._build_standard_prompt(task, motion_score, detected_objects)


# ============================================================================
# Step 2: Create Comparison Experiment
# ============================================================================

def compare_prompting_architectures(video_name="video_01"):
    """
    Compare different prompting architectures on the same video
    """
    
    print("="*70)
    print("PROMPTING ARCHITECTURE COMPARISON")
    print("="*70)
    print()
    
    # Define architectures to test
    architectures = [
        {
            'name': 'Standard',
            'template': 'standard',
            'description': 'Direct classification prompt'
        },
        {
            'name': 'Detailed',
            'template': 'detailed',
            'description': 'Verbose prompt with context'
        },
        {
            'name': 'Minimal',
            'template': 'minimal',
            'description': 'Concise prompt'
        },
        {
            'name': 'ReAct',
            'template': 'react',
            'description': 'Reasoning + Acting architecture',
            'custom_builder': ReActPromptBuilder
        }
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\nTesting: {arch['name']}")
        print(f"Description: {arch['description']}")
        print("-" * 70)
        
        # Create parameters
        params = BatchParameters(
            prompt_template=arch['template'],
            config_name=f"prompt_comparison_{arch['name'].lower()}",
            experiment_id="prompting_architecture_study",
            
            # Consistent settings for fair comparison
            llm_provider="gemini",
            llm_model="gemini-2.0-flash-exp",
            enable_object_detection=True,
            include_motion_score=True,
            include_object_list=True
        )
        
        # Use custom builder if specified
        if 'custom_builder' in arch:
            # Note: In practice, you'd modify video_processor.py to use custom builder
            # For this example, we'll just note it
            print(f"  Using custom builder: {arch['custom_builder'].__name__}")
        
        try:
            # Process video
            outputs = process_video(video_name, params)
            
            results.append({
                'architecture': arch['name'],
                'batch_id': params.batch_id,
                'success': True,
                'outputs': outputs
            })
            
            print(f"  ✓ Success - Batch ID: {params.batch_id}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'architecture': arch['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print()
    print("="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for result in results:
        if result['success']:
            print(f"✓ {result['architecture']}: {result['batch_id']}")
        else:
            print(f"✗ {result['architecture']}: Failed")
    
    print()
    print("Next steps:")
    print("1. Run accuracy benchmark on each batch_id")
    print("2. Compare results using compare_models.py")
    print("3. Analyze which architecture performs best")
    
    return results


# ============================================================================
# Step 3: Advanced Example - Adaptive Prompting
# ============================================================================

class AdaptivePromptBuilder(PromptBuilder):
    """
    Adaptive prompting: Choose strategy based on context
    """
    
    def build_action_classification_prompt(
        self,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Adapt prompt based on available information"""
        
        # High confidence scenario - use minimal prompt
        if motion_score is not None and detected_objects and len(detected_objects) >= 3:
            return self._build_minimal_prompt(
                'action_classification',
                motion_score,
                detected_objects
            )
        
        # Low confidence scenario - use detailed reasoning
        elif motion_score is None or not detected_objects:
            return self._build_react_prompt(
                'action_classification',
                motion_score,
                detected_objects
            )
        
        # Default
        else:
            return self._build_standard_prompt(
                'action_classification',
                motion_score,
                detected_objects
            )
    
    def _build_react_prompt(self, task, motion_score=None, detected_objects=None):
        """ReAct prompt for uncertain cases"""
        # Similar to ReActPromptBuilder above
        return "ReAct prompt here..."


# ============================================================================
# Step 4: Few-Shot Learning Example
# ============================================================================

class FewShotPromptBuilder(PromptBuilder):
    """
    Few-shot prompting: Provide examples
    """
    
    def _build_few_shot_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Build prompt with examples"""
        
        if task == 'action_classification':
            actions = ", ".join(self.batch_params.allowed_actions)
            
            examples = """Here are some examples:

Example 1:
- Motion: 0.85 (high)
- Objects: hammer, nail
- Classification: using tool

Example 2:
- Motion: 0.05 (low)
- Objects: none
- Classification: idle

Example 3:
- Motion: 0.45 (moderate)
- Objects: none
- Classification: moving

Now classify this scene:
"""
            
            current = f"- Motion: {motion_score:.2f}\n"
            if detected_objects:
                objs = ", ".join([obj[0] for obj in detected_objects[:3]])
                current += f"- Objects: {objs}\n"
            else:
                current += "- Objects: none\n"
            
            current += f"- Classification: [Choose from: {actions}]"
            
            return examples + current
        
        return super()._build_standard_prompt(task, motion_score, detected_objects)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Example: Custom Prompting Architectures")
    print()
    print("This example demonstrates:")
    print("1. Adding ReAct prompting architecture")
    print("2. Comparing multiple architectures")
    print("3. Adaptive prompting based on context")
    print("4. Few-shot learning")
    print()
    
    # Uncomment to run comparison
    # results = compare_prompting_architectures()
    
    print("To run the comparison:")
    print("  python example_custom_prompting.py")
    print()
    print("To add your own architecture:")
    print("  1. Create a class extending PromptBuilder")
    print("  2. Add your prompt building method")
    print("  3. Test with BatchParameters(prompt_template='your_template')")
    print("  4. Compare results using benchmarking tools")
