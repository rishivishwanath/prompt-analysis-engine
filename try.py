import json
import os
import statistics
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import time

# You'll need to install: pip install google-generativeai numpy
import google.generativeai as genai

# Configuration
DIMENSIONS = [
    "clarity",
    "specificity", 
    "context",
    "format",
    "tone_persona",
    "completeness",
    "conciseness"
]

DIMENSION_LABELS = {
    "clarity": "Clarity",
    "specificity": "Specificity",
    "context": "Context", 
    "format": "Format",
    "tone_persona": "Tone / Persona",
    "completeness": "Completeness",
    "conciseness": "Conciseness"
}

# Enhanced configuration for deterministic evaluation
EVALUATION_CONFIG = {
    "num_runs": 5,  # Number of evaluation runs for aggregation
    "temperature": 0.0,  # Lowest temperature for determinism
    "top_p": 0.1,  # Low top_p for more deterministic sampling
    "top_k": 1,  # Most deterministic sampling
    "max_output_tokens": 2048,
    "outlier_threshold": 2,  # Z-score threshold for outlier detection
    "delay_between_runs": 1,  # Seconds between API calls
}

@dataclass
class EvaluationStats:
    """Statistics for evaluation scores"""
    mean: float
    median: float
    std_dev: float
    min_score: float
    max_score: float
    confidence_interval: Tuple[float, float]
    outliers_removed: int

def load_conversation_from_file(file_path: str) -> List[Dict[str, str]]:
    """Load conversation from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('conversation', [])

def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """Format conversation for evaluation"""
    formatted_messages = []
    for message in conversation:
        role = "Candidate" if message["sender_type"] == "human" else "AI Assistant"
        formatted_messages.append(f"{role}: {message['message']}")
    return "\n".join(formatted_messages)

def create_evaluation_prompt(conversation_text: str, run_number: int = 1) -> str:
    """Create the evaluation prompt for the LLM with enhanced determinism instructions"""
    return f"""
You are a Prompt Quality Evaluator. This is evaluation run #{run_number}. 

CRITICAL INSTRUCTIONS FOR CONSISTENCY:
- Be extremely precise and consistent in your scoring
- Use the same evaluation criteria across all dimensions
- Base scores strictly on objective evidence from the conversation
- Avoid subjective interpretations that might vary between runs

CONVERSATION:
{conversation_text}

Evaluate the candidate's prompting skills based on the following metrics:
1. Clarity (1-10): Is it unambiguous and easy to understand?
2. Specificity (1-10): Does it define a clear task or goal?
3. Context (1-10): Does it provide background or assumptions?
4. Format (1-10): Does it specify the desired format of the output?
5. Tone/Persona (1-10): Is the desired tone or speaker persona defined?
6. Completeness (1-10): Is all necessary info provided?
7. Conciseness (1-10): Is it concise without missing key info?

For interaction quality, assess:
1. Iteration quality (1-10): How well did the candidate refine prompts based on AI responses?
2. Adaptation (1-10): How effectively did the candidate adjust when initial prompts weren't successful?
3. Follow-up effectiveness (1-10): Did the candidate ask appropriate clarifying questions?

SCORING GUIDELINES:
- 1-3: Poor/Absent
- 4-6: Fair/Basic
- 7-8: Good/Solid
- 9-10: Excellent/Exceptional

IMPORTANT: Return your response as a valid JSON object with this exact structure:
{{
  "clarity": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "specificity": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "context": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "format": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "tone_persona": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "completeness": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "conciseness": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "iteration_quality": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "adaptation": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "follow_up_effectiveness": {{"score": X, "reason": "detailed explanation with specific evidence"}},
  "total_score": X,
  "percentage_score": "XX%",
  "overall_feedback": "comprehensive feedback about the candidate's prompting abilities"
}}

Replace X with actual numeric scores. Ensure the JSON is properly formatted and valid.
"""

def initialize_gemini(api_key: str = None) -> genai.GenerativeModel:
    """Initialize Gemini with deterministic settings"""
    if api_key:
        genai.configure(api_key=api_key)
    else:
        genai.configure(api_key="AIzaSyB5LijGTV0692DQauzD1da9v3qY1B5wggQ")
    
    return genai.GenerativeModel('gemini-1.5-flash')

def get_deterministic_generation_config() -> genai.types.GenerationConfig:
    """Get generation config optimized for deterministic results"""
    return genai.types.GenerationConfig(
        temperature=EVALUATION_CONFIG["temperature"],
        top_p=EVALUATION_CONFIG["top_p"],
        top_k=EVALUATION_CONFIG["top_k"],
        max_output_tokens=EVALUATION_CONFIG["max_output_tokens"]
    )

def clean_json_response(response_text: str) -> str:
    """Clean the response text to extract valid JSON"""
    # Remove markdown code blocks if present
    if '```json' in response_text:
        start = response_text.find('```json') + 7
        end = response_text.find('```', start)
        response_text = response_text[start:end].strip()
    elif '```' in response_text:
        start = response_text.find('```') + 3
        end = response_text.find('```', start)
        response_text = response_text[start:end].strip()
    
    # Find JSON object boundaries
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    
    if start_idx != -1 and end_idx != -1:
        return response_text[start_idx:end_idx]
    
    return response_text

def parse_evaluation_response(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response and extract JSON"""
    try:
        cleaned_response = clean_json_response(response_text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Cleaned response: {cleaned_response[:200]}...")
        return None

def detect_outliers(scores: List[float], threshold: float = 2.0) -> List[bool]:
    """Detect outliers using Z-score method"""
    if len(scores) < 3:
        return [False] * len(scores)
    
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
    
    if std_dev == 0:
        return [False] * len(scores)
    
    z_scores = [(score - mean_score) / std_dev for score in scores]
    return [abs(z) > threshold for z in z_scores]

def calculate_stats(scores: List[float]) -> EvaluationStats:
    """Calculate comprehensive statistics for scores"""
    if not scores:
        return EvaluationStats(0, 0, 0, 0, 0, (0, 0), 0)
    
    # Remove outliers
    outlier_mask = detect_outliers(scores, EVALUATION_CONFIG["outlier_threshold"])
    clean_scores = [score for score, is_outlier in zip(scores, outlier_mask) if not is_outlier]
    outliers_removed = len(scores) - len(clean_scores)
    
    if not clean_scores:
        clean_scores = scores  # Use original if all are outliers
        outliers_removed = 0
    
    mean_score = statistics.mean(clean_scores)
    median_score = statistics.median(clean_scores)
    std_dev = statistics.stdev(clean_scores) if len(clean_scores) > 1 else 0
    min_score = min(clean_scores)
    max_score = max(clean_scores)
    
    # Calculate 95% confidence interval
    if len(clean_scores) > 1:
        margin_of_error = 1.96 * (std_dev / np.sqrt(len(clean_scores)))
        ci_lower = mean_score - margin_of_error
        ci_upper = mean_score + margin_of_error
    else:
        ci_lower = ci_upper = mean_score
    
    return EvaluationStats(
        mean=round(mean_score, 2),
        median=round(median_score, 2),
        std_dev=round(std_dev, 2),
        min_score=round(min_score, 2),
        max_score=round(max_score, 2),
        confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
        outliers_removed=outliers_removed
    )

def run_single_evaluation(model: genai.GenerativeModel, conversation_text: str, run_number: int) -> Dict[str, Any]:
    """Run a single evaluation"""
    try:
        prompt = create_evaluation_prompt(conversation_text, run_number)
        
        response = model.generate_content(
            prompt,
            generation_config=get_deterministic_generation_config()
        )
        
        result = parse_evaluation_response(response.text)
        if result:
            result["run_number"] = run_number
            return result
        else:
            print(f"Failed to parse response for run {run_number}")
            return None
            
    except Exception as e:
        print(f"Error in run {run_number}: {e}")
        return None

def aggregate_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple evaluation results with statistics"""
    if not results:
        return create_fallback_response()
    
    # Extract scores for each dimension
    all_dimensions = DIMENSIONS + ["iteration_quality", "adaptation", "follow_up_effectiveness"]
    aggregated = {}
    
    for dim in all_dimensions:
        scores = []
        reasons = []
        
        for result in results:
            if dim in result and isinstance(result[dim], dict):
                score = result[dim].get("score", 0)
                reason = result[dim].get("reason", "")
                if isinstance(score, (int, float)) and 0 <= score <= 10:
                    scores.append(float(score))
                    reasons.append(reason)
        
        if scores:
            stats = calculate_stats(scores)
            # Use median as the final score for better stability
            final_score = stats.median
            
            # Combine reasons (use the most common one or first one)
            final_reason = reasons[0] if reasons else "No reason provided"
            
            aggregated[dim] = {
                "score": final_score,
                "reason": final_reason,
                "stats": {
                    "mean": stats.mean,
                    "median": stats.median,
                    "std_dev": stats.std_dev,
                    "min": stats.min_score,
                    "max": stats.max_score,
                    "confidence_interval": stats.confidence_interval,
                    "outliers_removed": stats.outliers_removed,
                    "all_scores": scores
                }
            }
        else:
            aggregated[dim] = {
                "score": 0,
                "reason": "Failed to extract valid scores",
                "stats": None
            }
    
    # Calculate total scores
    total_scores = []
    for result in results:
        total = sum(
            result[dim].get("score", 0) 
            for dim in all_dimensions 
            if dim in result and isinstance(result[dim], dict)
        )
        if total > 0:
            total_scores.append(total)
    
    if total_scores:
        total_stats = calculate_stats(total_scores)
        final_total = total_stats.median
        final_percentage = f"{(final_total/100)*100:.0f}%"
    else:
        final_total = 0
        final_percentage = "0%"
        total_stats = None
    
    # Aggregate overall feedback
    feedback_list = [r.get("overall_feedback", "") for r in results if r.get("overall_feedback")]
    final_feedback = feedback_list[0] if feedback_list else "No feedback available"
    
    aggregated.update({
        "total_score": final_total,
        "percentage_score": final_percentage,
        "overall_feedback": final_feedback,
        "evaluation_metadata": {
            "num_runs": len(results),
            "successful_runs": len([r for r in results if r]),
            "total_stats": {
                "mean": total_stats.mean if total_stats else 0,
                "median": total_stats.median if total_stats else 0,
                "std_dev": total_stats.std_dev if total_stats else 0,
                "confidence_interval": total_stats.confidence_interval if total_stats else (0, 0),
                "all_scores": total_scores
            } if total_stats else None
        }
    })
    
    return aggregated

def evaluate_conversation_multi_shot(conversation_data: List[Dict[str, str]], api_key: str = None) -> Dict[str, Any]:
    """
    Evaluate a conversation multiple times and aggregate results for consistency
    """
    try:
        print("Initializing Gemini with deterministic settings...")
        model = initialize_gemini(api_key)
        
        print("Formatting conversation...")
        conversation_text = format_conversation(conversation_data)
        
        print(f"Running {EVALUATION_CONFIG['num_runs']} evaluation rounds for consistency...")
        
        results = []
        for run in range(1, EVALUATION_CONFIG['num_runs'] + 1):
            print(f"  Running evaluation {run}/{EVALUATION_CONFIG['num_runs']}...")
            
            result = run_single_evaluation(model, conversation_text, run)
            if result:
                results.append(result)
            
            # Delay between runs to avoid rate limiting
            if run < EVALUATION_CONFIG['num_runs']:
                time.sleep(EVALUATION_CONFIG['delay_between_runs'])
        
        if not results:
            print("All evaluation runs failed!")
            return create_fallback_response()
        
        print(f"Successfully completed {len(results)}/{EVALUATION_CONFIG['num_runs']} evaluation runs")
        print("Aggregating results with statistical analysis...")
        
        return aggregate_evaluation_results(results)
        
    except Exception as e:
        print(f"Error during multi-shot evaluation: {e}")
        return create_fallback_response()

def create_fallback_response() -> Dict[str, Any]:
    """Create a fallback response if evaluation fails"""
    return {
        "clarity": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "specificity": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "context": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "format": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "tone_persona": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "completeness": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "conciseness": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "iteration_quality": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "adaptation": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "follow_up_effectiveness": {"score": 0, "reason": "Failed to parse evaluation", "stats": None},
        "total_score": 0,
        "percentage_score": "0%",
        "overall_feedback": "Evaluation failed due to parsing error",
        "evaluation_metadata": {"num_runs": 0, "successful_runs": 0, "total_stats": None}
    }

def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results with statistical information"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("ENHANCED PROMPT QUALITY EVALUATION RESULTS")
    print("="*80)
    
    # Print metadata
    metadata = results.get("evaluation_metadata", {})
    print(f"\nEvaluation Metadata:")
    print(f"  Number of runs: {metadata.get('num_runs', 'N/A')}")
    print(f"  Successful runs: {metadata.get('successful_runs', 'N/A')}")
    
    total_stats = metadata.get("total_stats")
    if total_stats:
        print(f"  Total score std dev: {total_stats.get('std_dev', 'N/A')}")
        print(f"  Score consistency: {'High' if total_stats.get('std_dev', 999) < 2 else 'Medium' if total_stats.get('std_dev', 999) < 5 else 'Low'}")
    
    # Print individual dimension scores with statistics
    print("\nDIMENSION SCORES (with statistical analysis):")
    print("-" * 60)
    
    for dim in DIMENSIONS:
        if dim in results and isinstance(results[dim], dict):
            label = DIMENSION_LABELS.get(dim, dim.title())
            score_data = results[dim]
            stats = score_data.get("stats")
            
            print(f"\n{label}: {score_data.get('score', 'N/A')}/10")
            print(f"  Reason: {score_data.get('reason', 'N/A')}")
            
            if stats:
                print(f"  Statistics:")
                print(f"    Mean: {stats['mean']}, Median: {stats['median']}")
                print(f"    Std Dev: {stats['std_dev']} (Range: {stats['min']}-{stats['max']})")
                print(f"    95% CI: {stats['confidence_interval']}")
                if stats['outliers_removed'] > 0:
                    print(f"    Outliers removed: {stats['outliers_removed']}")
    
    # Print interaction quality scores
    print("\n" + "-" * 60)
    print("INTERACTION QUALITY (with statistical analysis):")
    print("-" * 60)
    
    interaction_dims = ["iteration_quality", "adaptation", "follow_up_effectiveness"]
    interaction_labels = {
        "iteration_quality": "Iteration Quality",
        "adaptation": "Adaptation",
        "follow_up_effectiveness": "Follow-up Effectiveness"
    }
    
    for dim in interaction_dims:
        if dim in results and isinstance(results[dim], dict):
            label = interaction_labels.get(dim, dim.replace("_", " ").title())
            score_data = results[dim]
            stats = score_data.get("stats")
            
            print(f"\n{label}: {score_data.get('score', 'N/A')}/10")
            print(f"  Reason: {score_data.get('reason', 'N/A')}")
            
            if stats:
                print(f"  Statistics:")
                print(f"    Mean: {stats['mean']}, Median: {stats['median']}")
                print(f"    Std Dev: {stats['std_dev']} (Range: {stats['min']}-{stats['max']})")
                print(f"    95% CI: {stats['confidence_interval']}")
                if stats['outliers_removed'] > 0:
                    print(f"    Outliers removed: {stats['outliers_removed']}")
    
    # Print overall scores
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT:")
    print("="*60)
    print(f"Final Score: {results.get('total_score', 'N/A')}/100")
    print(f"Percentage: {results.get('percentage_score', 'N/A')}")
    
    if total_stats:
        print(f"Score Range: {total_stats.get('confidence_interval', 'N/A')}")
        print(f"Consistency Rating: {'High' if total_stats.get('std_dev', 999) < 2 else 'Medium' if total_stats.get('std_dev', 999) < 5 else 'Low'}")
    
    print(f"\nOverall Feedback:")
    print(f"{results.get('overall_feedback', 'N/A')}")
    print("="*80)

def save_results_to_file(results: Dict[str, Any], filename: str = "enhanced_evaluation_results.json"):
    """Save evaluation results to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to run the enhanced evaluation"""
    print("Starting Enhanced Deterministic Prompt Quality Evaluation...")
    print(f"Configuration: {EVALUATION_CONFIG['num_runs']} runs, temp={EVALUATION_CONFIG['temperature']}")
    
    # Load conversation from conversation.json file
    conversation_file = "conversation1.json"
    
    try:
        print(f"Loading conversation from {conversation_file}...")
        conversation_data = load_conversation_from_file(conversation_file)
        print(f"Loaded conversation with {len(conversation_data)} messages")
        
        # Run multi-shot evaluation
        print("\nRunning multi-shot evaluation for enhanced consistency...")
        results = evaluate_conversation_multi_shot(conversation_data)
        
        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_results_to_file(results)
        
        print("\nEnhanced evaluation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: {conversation_file} not found in the current directory.")
        print("Please make sure conversation.json exists in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

# To use this enhanced script:
# 1. Install: pip install google-generativeai numpy
# 2. Make sure conversation.json exists in the same directory
# 3. Run: python enhanced_evaluator.py
#
# Key improvements for deterministic behavior:
# - Multiple evaluation runs with statistical aggregation
# - Outlier detection and removal
# - Confidence intervals for score reliability
# - Enhanced deterministic generation settings
# - Comprehensive statistical reporting