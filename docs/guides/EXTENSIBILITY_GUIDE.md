# Extensibility Guide - Adding New Methods & Architectures

A comprehensive guide to extending the video processing system with custom methods, prompting architectures, and experimental features.

---

## Table of Contents

1. [Overview](#overview)
2. [Adding Custom Prompt Templates](#adding-custom-prompt-templates)
3. [Adding New Action Classification Methods](#adding-new-action-classification-methods)
4. [Adding New LLM Providers](#adding-new-llm-providers)
5. [Adding New CV Models](#adding-new-cv-models)
6. [Creating Custom Analysis Pipelines](#creating-custom-analysis-pipelines)
7. [Experimenting with Prompting Architectures](#experimenting-with-prompting-architectures)
8. [Best Practices](#best-practices)

---

## Overview

The system uses a **registry pattern** that makes it easy to add new methods without modifying core code. Key extension points:

- **Prompt Templates** - Add new prompting strategies
- **Action Classifiers** - Add new classification methods
- **Tool Detectors** - Add new tool detection strategies
- **LLM Providers** - Add new AI models
- **CV Models** - Add new computer vision models

**Philosophy**: Extend, don't modify. Add new methods alongside existing ones.

---

## Adding Custom Prompt Templates

### Method 1: Use Built-in Custom Template

**Easiest approach** - Just provide a template file:

```python
from video_processing import BatchParameters

params = BatchParameters(
    prompt_template="custom",
    custom_prompt_path="my_prompts/advanced_prompt.txt"
)
```

**Template file format** (`my_prompts/advanced_prompt.txt`):
```
You are analyzing a construction site video.

Context:
- Motion score: {motion_score}
- Detected objects: {objects}

Task: {task}

Available actions: {actions}
Available tools: {tools}

Respond with ONE word only.
```

**Placeholders**:
- `{task}` - "action_classification" or "tool_detection"
- `{actions}` - Comma-separated allowed actions
- `{tools}` - Comma-separated allowed tools
- `{motion_score}` - Motion score (if enabled)
- `{objects}` - Detected objects (if enabled)

### Method 2: Add New Template to PromptBuilder

**For reusable templates** - Add to the system:

**File**: `video_processing/ai/prompt_builder.py`

```python
class PromptBuilder:
    def __init__(self, batch_params):
        self.batch_params = batch_params
        self.prompt_templates = {
            'standard': self._build_standard_prompt,
            'detailed': self._build_detailed_prompt,
            'minimal': self._build_minimal_prompt,
            'custom': self._build_custom_prompt,
            'chain_of_thought': self._build_cot_prompt,  # NEW!
        }
    
    def _build_cot_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Chain-of-thought prompting for better reasoning"""
        
        if task == 'action_classification':
            actions = ", ".join(self.batch_params.allowed_actions)
            
            prompt = (
                f"Analyze this construction site video step by step:\n\n"
                f"Step 1: Observe what you see in the frames.\n"
                f"Step 2: Identify any tools or equipment.\n"
                f"Step 3: Determine the worker's activity.\n\n"
            )
            
            if motion_score is not None:
                prompt += f"Motion level: {motion_score:.2f}\n"
            
            if detected_objects:
                objects_str = ", ".join([obj[0] for obj in detected_objects[:5]])
                prompt += f"Detected objects: {objects_str}\n"
            
            prompt += (
                f"\nStep 4: Classify the activity as ONE of: {actions}\n"
                f"Respond with ONLY the category name."
            )
            
            return prompt
        
        # Similar for tool_detection...
        return self._build_standard_prompt(task, motion_score, detected_objects)
```

**Usage**:
```python
from video_processing import BatchParameters, PromptTemplate

params = BatchParameters(
    prompt_template="chain_of_thought"  # Use new template
)
```

### Method 3: Dynamic Prompt Generation

**For experimental architectures** - Generate prompts programmatically:

```python
class AdaptivePromptBuilder(PromptBuilder):
    """Adapts prompts based on context"""
    
    def build_action_classification_prompt(self, motion_score=None, detected_objects=None):
        # Choose template based on context
        if motion_score and motion_score > 0.5:
            # High motion - use action-focused prompt
            return self._build_action_focused_prompt(motion_score, detected_objects)
        elif detected_objects and len(detected_objects) > 3:
            # Many objects - use object-focused prompt
            return self._build_object_focused_prompt(detected_objects)
        else:
            # Default
            return self._build_standard_prompt('action_classification', motion_score, detected_objects)
```

---

## Adding New Action Classification Methods

The system uses a **registry pattern** for action classifiers. Add new methods easily:

### Step 1: Define Your Method

**File**: `video_processing/analysis/action_classifier.py`

```python
# Add to the registry
_action_classifiers = {}

def register_action_classifier(name: str):
    """Decorator to register action classification methods"""
    def decorator(func):
        _action_classifiers[name] = func
        return func
    return decorator

# Your new method
@register_action_classifier("multimodal_fusion")
def classify_action_multimodal(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Multi-modal fusion: Combines visual, motion, and object features
    """
    # Extract visual features
    visual_features = extract_visual_features(frames)
    
    # Combine with motion
    if motion_score:
        combined_features = fuse_features(visual_features, motion_score)
    
    # Use LLM for final classification
    prompt = f"Based on these features: {combined_features}, classify as: {batch_params.allowed_actions}"
    
    response = llm_service.send_multiframe_prompt(
        frames=frames,
        prompt_text=prompt,
        max_tokens=100,
        temperature=0.0
    )
    
    return response.strip().lower()

def extract_visual_features(frames):
    """Extract features from frames"""
    # Your feature extraction logic
    return {"avg_brightness": 0.5, "edge_density": 0.3}

def fuse_features(visual, motion):
    """Combine features"""
    return {"visual": visual, "motion": motion}
```

### Step 2: Add to BatchParameters

**File**: `video_processing/batch_parameters.py`

```python
class ActionClassificationMethod(str, Enum):
    LLM_ONLY = "llm_only"
    MOTION_AWARE = "motion_aware"
    OBJECT_AWARE = "object_aware"
    MULTIMODAL_FUSION = "multimodal_fusion"  # NEW!
    CUSTOM = "custom"
```

### Step 3: Use Your Method

```python
from video_processing import BatchParameters, ActionClassificationMethod

params = BatchParameters(
    action_classification_method=ActionClassificationMethod.MULTIMODAL_FUSION,
    config_name="multimodal_experiment"
)

outputs = process_video("video_01", params)
```

### Step 4: Compare Methods

```python
# Test different classification methods
methods = [
    ActionClassificationMethod.LLM_ONLY,
    ActionClassificationMethod.MOTION_AWARE,
    ActionClassificationMethod.MULTIMODAL_FUSION
]

for method in methods:
    params = BatchParameters(
        action_classification_method=method,
        experiment_id="method_comparison"
    )
    outputs = process_video("video_01", params)
    print(f"Batch ID for {method}: {params.batch_id}")
```

---

## Adding New LLM Providers

### Step 1: Create Provider Class

**File**: `video_processing/ai/llm_service.py`

```python
class CustomLLMService(LLMService):
    """Your custom LLM provider"""
    
    def __init__(self, api_key: str, model: str = "custom-model-v1"):
        # Initialize your client
        from your_llm_library import Client
        self.client = Client(api_key=api_key)
        self.model = model
    
    def send_multiframe_prompt(
        self,
        frames: List[np.ndarray],
        prompt_text: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Send frames to your LLM"""
        
        # Encode frames
        encoded_frames = self._encode_frames(frames)
        
        # Call your API
        response = self.client.generate(
            model=self.model,
            images=encoded_frames,
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.text.strip()
    
    def _encode_frames(self, frames: List[np.ndarray]) -> List[str]:
        """Encode frames for your API"""
        # Your encoding logic
        return [encode_frame(f) for f in frames]
    
    def get_provider_name(self) -> str:
        return f"custom_{self.model}"
```

### Step 2: Add to Factory Function

```python
def get_llm_service(batch_params) -> LLMService:
    """Factory function for LLM services"""
    
    provider = batch_params.llm_provider
    
    if provider == LLMProvider.CLAUDE:
        return ClaudeService(...)
    elif provider == LLMProvider.GEMINI:
        return GeminiService(...)
    elif provider == LLMProvider.CUSTOM:  # NEW!
        return CustomLLMService(
            api_key=batch_params.llm_api_key,
            model=batch_params.llm_model
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### Step 3: Use Your Provider

```python
params = BatchParameters(
    llm_provider="custom",  # Or add to LLMProvider enum
    llm_model="custom-model-v1",
    llm_api_key="your-api-key"
)
```

---

## Adding New CV Models

### Step 1: Create CV Service

**File**: `video_processing/ai/cv_service.py`

```python
class CustomCVService(CVService):
    """Your custom computer vision model"""
    
    def __init__(self, model_path: str):
        from your_cv_library import load_model
        self.model = load_model(model_path)
    
    def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Detect objects in frame"""
        
        # Run your model
        results = self.model.predict(frame)
        
        # Format results
        detections = []
        for det in results:
            if det.confidence >= confidence_threshold:
                detections.append((
                    det.class_name,
                    det.confidence,
                    (det.x1, det.y1, det.x2, det.y2)
                ))
        
        return detections
    
    def get_model_name(self) -> str:
        return "custom_cv_model"
```

### Step 2: Add to Factory

```python
def get_cv_service(batch_params) -> CVService:
    """Factory for CV services"""
    
    if batch_params.cv_model == CVModel.CUSTOM:
        return CustomCVService(batch_params.cv_model_path)
    # ... existing models
```

---

## Creating Custom Analysis Pipelines

### Example: Multi-Stage Analysis

```python
from video_processing import BatchParameters, process_video

class MultiStageAnalyzer:
    """Custom multi-stage analysis pipeline"""
    
    def __init__(self):
        self.stages = []
    
    def add_stage(self, stage_params):
        """Add analysis stage"""
        self.stages.append(stage_params)
    
    def run(self, video_name):
        """Run all stages"""
        results = []
        
        for i, params in enumerate(self.stages):
            print(f"Stage {i+1}: {params.config_name}")
            outputs = process_video(video_name, params)
            results.append({
                'stage': i+1,
                'batch_id': params.batch_id,
                'outputs': outputs
            })
        
        return results

# Usage
analyzer = MultiStageAnalyzer()

# Stage 1: Quick pass with basic classification
analyzer.add_stage(BatchParameters(
    config_name="stage1_quick",
    llm_model="gemini-2.0-flash-exp",  # Fast model
    enable_object_detection=False
))

# Stage 2: Detailed analysis on interesting frames
analyzer.add_stage(BatchParameters(
    config_name="stage2_detailed",
    llm_model="claude-sonnet-4",  # Better model
    enable_object_detection=True,
    enable_relationship_tracking=True
))

results = analyzer.run("video_01")
```

---

## Experimenting with Prompting Architectures

### 1. Few-Shot Prompting

```python
class FewShotPromptBuilder(PromptBuilder):
    """Add examples to prompts"""
    
    def _build_few_shot_prompt(self, task, motion_score=None, detected_objects=None):
        examples = """
        Example 1:
        Scene: Person holding hammer, motion score 0.8
        Classification: using tool
        
        Example 2:
        Scene: Person standing still, motion score 0.1
        Classification: idle
        
        Example 3:
        Scene: Person walking, motion score 0.6
        Classification: moving
        """
        
        prompt = examples + "\n\nNow classify this scene:\n"
        prompt += f"Motion score: {motion_score}\n"
        prompt += f"Objects: {detected_objects}\n"
        prompt += "Classification: "
        
        return prompt
```

### 2. Chain-of-Thought

```python
def build_cot_prompt(self, task, motion_score=None, detected_objects=None):
    """Encourage step-by-step reasoning"""
    return """
    Let's analyze this step by step:
    
    1. What objects do you see? {objects}
    2. Is there motion? {motion_score}
    3. What is the person doing?
    4. Which category best fits?
    
    Think through each step, then provide your final answer.
    """
```

### 3. Self-Consistency

```python
class SelfConsistencyClassifier:
    """Sample multiple times and vote"""
    
    def classify(self, frames, params, llm_service, prompt_builder):
        # Generate same prompt multiple times with temperature > 0
        votes = []
        for _ in range(5):
            response = llm_service.send_multiframe_prompt(
                frames=frames,
                prompt_text=prompt_builder.build_action_classification_prompt(),
                temperature=0.7  # Non-zero for diversity
            )
            votes.append(response.strip())
        
        # Return most common answer
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]
```

### 4. Prompt Ensembling

```python
def ensemble_prompts(frames, params, llm_service):
    """Use multiple prompt strategies and combine"""
    
    prompts = [
        "Classify this construction activity: {actions}",
        "What is the worker doing? Choose from: {actions}",
        "Identify the action. Options: {actions}"
    ]
    
    results = []
    for prompt_template in prompts:
        prompt = prompt_template.format(actions=params.allowed_actions)
        response = llm_service.send_multiframe_prompt(frames, prompt)
        results.append(response.strip())
    
    # Vote or combine
    from collections import Counter
    return Counter(results).most_common(1)[0][0]
```

---

## Best Practices

### 1. Use Batch IDs for Tracking

```python
# Always save your experimental configs
params = BatchParameters(
    config_name="my_experiment",
    experiment_id="prompting_study_2025"
)
params.save_batch_config()

# Process
outputs = process_video("video_01", params)

# Track the batch_id
print(f"Experiment batch: {params.batch_id}")
```

### 2. Compare Systematically

```python
# Test multiple approaches
approaches = [
    ("standard", "standard"),
    ("detailed", "detailed"),
    ("cot", "chain_of_thought"),
    ("few_shot", "few_shot")
]

for name, template in approaches:
    params = BatchParameters(
        prompt_template=template,
        config_name=f"prompt_comparison_{name}",
        experiment_id="prompt_study"
    )
    process_video("video_01", params)

# Then use benchmarking to compare
from post_processing.model_comparison import compare_batches
# ... compare results
```

### 3. Document Your Methods

```python
@register_action_classifier("my_method")
def classify_my_way(frames, batch_params, llm_service, prompt_builder, **kwargs):
    """
    My custom classification method.
    
    Approach:
    - Uses multi-scale feature extraction
    - Combines with temporal context
    - Applies ensemble voting
    
    Best for:
    - Videos with complex activities
    - When accuracy > speed
    
    Parameters:
    - Set max_workers_keyframes=1 for sequential processing
    - Use temperature=0.0 for deterministic results
    """
    # Your implementation
    pass
```

### 4. Version Your Experiments

```python
params = BatchParameters(
    config_name="multimodal_v2",
    experiment_id="2025_01_prompting_study",
    model_version="1.2.3",  # Track your method version
    notes="Testing chain-of-thought with 5-shot examples"
)
```

---

## Complete Example: Adding a New Prompting Architecture

Let's add a "ReAct" (Reasoning + Acting) prompting architecture:

### Step 1: Create the Prompt Builder

```python
# File: video_processing/ai/prompt_builder.py

def _build_react_prompt(self, task, motion_score=None, detected_objects=None):
    """ReAct: Reasoning and Acting prompting"""
    
    if task == 'action_classification':
        actions = ", ".join(self.batch_params.allowed_actions)
        
        prompt = f"""
You are analyzing a construction site video. Use this format:

Thought: What do I observe in these frames?
Observation: [Describe what you see]

Thought: What does the motion score tell me?
Observation: Motion score is {motion_score:.2f}

Thought: What objects are present?
Observation: {', '.join([obj[0] for obj in (detected_objects or [])])}

Thought: Based on all observations, what is the worker doing?
Action: [Choose ONE from: {actions}]

Provide your final Action ONLY.
"""
        return prompt
```

### Step 2: Register It

```python
# In PromptBuilder.__init__
self.prompt_templates = {
    # ... existing templates
    'react': self._build_react_prompt,
}
```

### Step 3: Test It

```python
from video_processing import BatchParameters

# Test ReAct prompting
params_react = BatchParameters(
    prompt_template="react",
    config_name="react_prompting_test"
)

outputs = process_video("video_01", params_react)
print(f"ReAct batch: {params_react.batch_id}")

# Compare with standard
params_standard = BatchParameters(
    prompt_template="standard",
    config_name="standard_prompting_test"
)

outputs = process_video("video_01", params_standard)
print(f"Standard batch: {params_standard.batch_id}")

# Benchmark the difference
from post_processing.accuracy_benchmark import run_benchmark
react_results = run_benchmark(params_react.batch_id)
standard_results = run_benchmark(params_standard.batch_id)
```

---

## Summary

The system is **highly extensible** through:

1. **Registry Pattern** - Add methods without modifying core code
2. **Factory Functions** - Easy integration of new providers
3. **Batch Tracking** - Automatic experiment tracking
4. **Modular Design** - Replace any component independently

**Key Files for Extension**:
- `prompt_builder.py` - Add prompting strategies
- `action_classifier.py` - Add classification methods
- `llm_service.py` - Add LLM providers
- `cv_service.py` - Add CV models
- `batch_parameters.py` - Add configuration options

**Next Steps**:
1. Try the examples above
2. Create your custom method
3. Use batch tracking to compare
4. Use benchmarking to evaluate

Happy experimenting! 🚀
