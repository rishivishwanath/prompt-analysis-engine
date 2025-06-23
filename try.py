import json
import os
import statistics
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import time
import sqlite3
from datetime import datetime
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# You'll need to install: pip install google-generativeai numpy scikit-learn
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

# Enhanced configuration for deterministic evaluation with RAG
EVALUATION_CONFIG = {
    "num_runs": 5,
    "temperature": 0.0,
    "top_p": 0.1,
    "top_k": 1,
    "max_output_tokens": 2048,
    "outlier_threshold": 2,
    "delay_between_runs": 1,
    # RAG-specific configuration
    "rag_enabled": True,
    "max_examples": 3,  # Maximum number of RAG examples to include
    "similarity_threshold": 0.85,  # Minimum similarity score for RAG retrieval
    "vectorizer_max_features": 1000,
    "conversation_db": "evaluation_history.db",
    "vectorizer_cache": "vectorizer_cache.pkl"
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

@dataclass
class RAGExample:
    """Container for RAG retrieved examples"""
    conversation_id: str
    conversation_text: str
    scores: Dict[str, float]
    overall_feedback: str
    similarity_score: float
    evaluation_date: str

class RAGVectorStore:
    """Vector store for conversation embeddings and retrieval"""
    
    def __init__(self, db_path: str = None, vectorizer_cache_path: str = None):
        self.db_path = db_path or EVALUATION_CONFIG["conversation_db"]
        self.vectorizer_cache_path = vectorizer_cache_path or EVALUATION_CONFIG["vectorizer_cache"]
        self.vectorizer = None
        self.conversation_vectors = None
        self.conversation_ids = []
        self._init_database()
        self._load_or_create_vectorizer()
    
    def _init_database(self):
        """Initialize SQLite database for storing evaluations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                conversation_text TEXT NOT NULL,
                conversation_hash TEXT NOT NULL,
                clarity_score REAL,
                specificity_score REAL,
                context_score REAL,
                format_score REAL,
                tone_persona_score REAL,
                completeness_score REAL,
                conciseness_score REAL,
                iteration_quality_score REAL,
                adaptation_score REAL,
                follow_up_effectiveness_score REAL,
                total_score REAL,
                percentage_score TEXT,
                overall_feedback TEXT,
                evaluation_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_or_create_vectorizer(self):
        """Load existing vectorizer or create new one"""
        try:
            if os.path.exists(self.vectorizer_cache_path):
                with open(self.vectorizer_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.vectorizer = cache_data['vectorizer']
                    self.conversation_vectors = cache_data['vectors']
                    self.conversation_ids = cache_data['conversation_ids']
                print(f"Loaded vectorizer cache with {len(self.conversation_ids)} conversations")
            else:
                self._rebuild_vectorizer()
        except Exception as e:
            print(f"Error loading vectorizer cache: {e}")
            self._rebuild_vectorizer()
    
    def _rebuild_vectorizer(self):
        """Rebuild vectorizer from database"""
        print("Rebuilding vectorizer from database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT conversation_id, conversation_text FROM evaluations")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("No existing conversations found. Creating empty vectorizer.")
            self.vectorizer = TfidfVectorizer(
                max_features=EVALUATION_CONFIG["vectorizer_max_features"],
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.conversation_vectors = None
            self.conversation_ids = []
            return
        
        self.conversation_ids = [row[0] for row in rows]
        conversation_texts = [row[1] for row in rows]
        
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=EVALUATION_CONFIG["vectorizer_max_features"],
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.conversation_vectors = self.vectorizer.fit_transform(conversation_texts)
        
        # Cache the vectorizer
        self._save_vectorizer_cache()
        print(f"Rebuilt vectorizer with {len(self.conversation_ids)} conversations")
    
    def _save_vectorizer_cache(self):
        """Save vectorizer and vectors to cache"""
        try:
            cache_data = {
                'vectorizer': self.vectorizer,
                'vectors': self.conversation_vectors,
                'conversation_ids': self.conversation_ids
            }
            with open(self.vectorizer_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving vectorizer cache: {e}")
    
    def store_evaluation(self, conversation_id: str, conversation_text: str, results: Dict[str, Any]):
        """Store evaluation results in database and update vectors"""
        conversation_hash = hashlib.sha256(conversation_text.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract scores from results
        scores = {}
        for dim in DIMENSIONS + ["iteration_quality", "adaptation", "follow_up_effectiveness"]:
            if dim in results and isinstance(results[dim], dict):
                scores[f"{dim}_score"] = results[dim].get("score", 0)
        
        # Insert or update evaluation
        cursor.execute('''
            INSERT OR REPLACE INTO evaluations (
                conversation_id, conversation_text, conversation_hash,
                clarity_score, specificity_score, context_score, format_score,
                tone_persona_score, completeness_score, conciseness_score,
                iteration_quality_score, adaptation_score, follow_up_effectiveness_score,
                total_score, percentage_score, overall_feedback, evaluation_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_id, conversation_text, conversation_hash,
            scores.get('clarity_score', 0), scores.get('specificity_score', 0),
            scores.get('context_score', 0), scores.get('format_score', 0),
            scores.get('tone_persona_score', 0), scores.get('completeness_score', 0),
            scores.get('conciseness_score', 0), scores.get('iteration_quality_score', 0),
            scores.get('adaptation_score', 0), scores.get('follow_up_effectiveness_score', 0),
            results.get('total_score', 0), results.get('percentage_score', '0%'),
            results.get('overall_feedback', ''), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Update vectorizer if this is a new conversation
        if conversation_id not in self.conversation_ids:
            self._add_conversation_to_vectorizer(conversation_id, conversation_text)
    
    def _add_conversation_to_vectorizer(self, conversation_id: str, conversation_text: str):
        """Add new conversation to vectorizer"""
        try:
            self.conversation_ids.append(conversation_id)
            
            if self.conversation_vectors is None:
                # First conversation - create initial vectors
                self.conversation_vectors = self.vectorizer.fit_transform([conversation_text])
            else:
                # Add to existing vectors
                new_vector = self.vectorizer.transform([conversation_text])
                self.conversation_vectors = np.vstack([self.conversation_vectors.toarray(), new_vector.toarray()])
                from scipy.sparse import csr_matrix
                self.conversation_vectors = csr_matrix(self.conversation_vectors)
            
            # Update cache
            self._save_vectorizer_cache()
            
        except Exception as e:
            print(f"Error adding conversation to vectorizer: {e}")
            # Rebuild vectorizer as fallback
            self._rebuild_vectorizer()
    
    def retrieve_similar_conversations(self, query_conversation: str, max_examples: int = 3) -> List[RAGExample]:
        """Retrieve similar conversations for RAG"""
        if not self.conversation_ids or self.conversation_vectors is None:
            return []
        
        try:
            # Vectorize query conversation
            query_vector = self.vectorizer.transform([query_conversation])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.conversation_vectors).flatten()
            
            # Get top similar conversations
            top_indices = np.argsort(similarities)[::-1][:max_examples]
            
            # Retrieve evaluation data from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            examples = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                # Skip if similarity is too low
                if similarity_score < EVALUATION_CONFIG["similarity_threshold"]:
                    continue
                
                conversation_id = self.conversation_ids[idx]
                
                cursor.execute('''
                    SELECT conversation_text, clarity_score, specificity_score, context_score,
                           format_score, tone_persona_score, completeness_score, conciseness_score,
                           iteration_quality_score, adaptation_score, follow_up_effectiveness_score,
                           overall_feedback, evaluation_date
                    FROM evaluations WHERE conversation_id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    scores = {
                        'clarity': row[1], 'specificity': row[2], 'context': row[3],
                        'format': row[4], 'tone_persona': row[5], 'completeness': row[6],
                        'conciseness': row[7], 'iteration_quality': row[8],
                        'adaptation': row[9], 'follow_up_effectiveness': row[10]
                    }
                    
                    examples.append(RAGExample(
                        conversation_id=conversation_id,
                        conversation_text=row[0],
                        scores=scores,
                        overall_feedback=row[11] or "",
                        similarity_score=similarity_score,
                        evaluation_date=row[12] or ""
                    ))
            
            conn.close()
            return examples
            
        except Exception as e:
            print(f"Error retrieving similar conversations: {e}")
            return []

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

def create_rag_examples_section(examples: List[RAGExample]) -> str:
    """Create the RAG examples section for the prompt"""
    if not examples:
        return ""
    
    examples_text = "\n\nPREVIOUS EVALUATION EXAMPLES (for consistency reference):\n"
    examples_text += "Use these as calibration examples to maintain scoring consistency:\n\n"
    
    for i, example in enumerate(examples, 1):
        examples_text += f"EXAMPLE {i} (Similarity: {example.similarity_score:.2f}):\n"
        examples_text += f"Conversation: {example.conversation_text[:300]}...\n"
        examples_text += "Scores:\n"
        
        for dim in DIMENSIONS:
            if dim in example.scores:
                label = DIMENSION_LABELS.get(dim, dim.title())
                examples_text += f"  {label}: {example.scores[dim]}/10\n"
        
        examples_text += f"  Iteration Quality: {example.scores.get('iteration_quality', 0)}/10\n"
        examples_text += f"  Adaptation: {example.scores.get('adaptation', 0)}/10\n"
        examples_text += f"  Follow-up Effectiveness: {example.scores.get('follow_up_effectiveness', 0)}/10\n"
        examples_text += f"Overall Feedback: {example.overall_feedback[:200]}...\n\n"
    
    examples_text += "IMPORTANT: Use these examples to calibrate your scoring. "
    examples_text += "Similar conversations should receive similar scores. "
    examples_text += "Maintain consistency with the demonstrated scoring patterns.\n"
    examples_text += "-" * 60 + "\n"
    
    return examples_text

def create_evaluation_prompt(conversation_text: str, rag_examples: List[RAGExample] = None, run_number: int = 1) -> str:
    """Create the evaluation prompt for the LLM with RAG examples"""
    rag_section = create_rag_examples_section(rag_examples) if rag_examples else ""
    
    return f"""
You are a Prompt Quality Evaluator. This is evaluation run #{run_number}. 

CRITICAL INSTRUCTIONS FOR CONSISTENCY:
- Be extremely precise and consistent in your scoring
- Use the same evaluation criteria across all dimensions
- Base scores strictly on objective evidence from the conversation
- Avoid subjective interpretations that might vary between runs
- If provided with previous examples, use them as calibration references for consistent scoring

{rag_section}

CONVERSATION TO EVALUATE:
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

CONSISTENCY REMINDER: If examples were provided above, ensure your scores align with similar conversation patterns shown in the examples.

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

def run_single_evaluation(model: genai.GenerativeModel, conversation_text: str, rag_examples: List[RAGExample], run_number: int) -> Dict[str, Any]:
    """Run a single evaluation with RAG examples"""
    try:
        prompt = create_evaluation_prompt(conversation_text, rag_examples, run_number)
        
        response = model.generate_content(
            prompt,
            generation_config=get_deterministic_generation_config()
        )
        
        result = parse_evaluation_response(response.text)
        if result:
            result["run_number"] = run_number
            result["rag_examples_used"] = len(rag_examples) if rag_examples else 0
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
    
    # RAG metadata
    rag_examples_used = results[0].get("rag_examples_used", 0) if results else 0
    
    aggregated.update({
        "total_score": final_total,
        "percentage_score": final_percentage,
        "overall_feedback": final_feedback,
        "evaluation_metadata": {
            "num_runs": len(results),
            "successful_runs": len([r for r in results if r]),
            "rag_examples_used": rag_examples_used,
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

def evaluate_conversation_with_rag(conversation_data: List[Dict[str, str]], api_key: str = None) -> Dict[str, Any]:
    """
    Evaluate a conversation with RAG-enhanced consistency
    """
    try:
        print("Initializing RAG-enhanced evaluation system...")
        
        # Initialize components
        model = initialize_gemini(api_key)
        vector_store = RAGVectorStore() if EVALUATION_CONFIG["rag_enabled"] else None
        
        print("Formatting conversation...")
        conversation_text = format_conversation(conversation_data)
        
        # Retrieve similar conversations for RAG
        rag_examples = []
        if vector_store and EVALUATION_CONFIG["rag_enabled"]:
            print("Retrieving similar conversations for RAG...")
            rag_examples = vector_store.retrieve_similar_conversations(
                conversation_text, 
                EVALUATION_CONFIG["max_examples"]
            )
            print(f"Found {len(rag_examples)} similar conversations for RAG")
            
            if rag_examples:
                print("RAG examples similarity scores:", [f"{ex.similarity_score:.3f}" for ex in rag_examples])
        
        print(f"Running {EVALUATION_CONFIG['num_runs']} evaluation rounds with RAG enhancement...")
        
        results = []
        for run in range(1, EVALUATION_CONFIG['num_runs'] + 1):
            print(f"  Running evaluation {run}/{EVALUATION_CONFIG['num_runs']}...")
            
            result = run_single_evaluation(model, conversation_text, rag_examples, run)
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
        
        final_results = aggregate_evaluation_results(results)
        
        # Store results in vector store for future RAG
        if vector_store and EVALUATION_CONFIG["rag_enabled"]:
            conversation_id = hashlib.sha256(conversation_text.encode()).hexdigest()[:16]
            print(f"Storing evaluation results for future RAG (ID: {conversation_id})")
            vector_store.store_evaluation(conversation_id, conversation_text, final_results)
        
        return final_results
        
    except Exception as e:
        print(f"Error during RAG-enhanced evaluation: {e}")
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
        "evaluation_metadata": {"num_runs": 0, "successful_runs": 0, "rag_examples_used": 0, "total_stats": None}
    }

def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results with statistical information and RAG details"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("RAG-ENHANCED PROMPT QUALITY EVALUATION RESULTS")
    print("="*80)
    
    # Print metadata
    metadata = results.get("evaluation_metadata", {})
    print(f"\nEvaluation Metadata:")
    print(f"  Number of runs: {metadata.get('num_runs', 'N/A')}")
    print(f"  Successful runs: {metadata.get('successful_runs', 'N/A')}")
    print(f"  RAG examples used: {metadata.get('rag_examples_used', 'N/A')}")
    
    total_stats = metadata.get("total_stats")
    if total_stats:
        print(f"  Total score std dev: {total_stats.get('std_dev', 'N/A')}")
        consistency_rating = 'High' if total_stats.get('std_dev', 999) < 2 else 'Medium' if total_stats.get('std_dev', 999) < 5 else 'Low'
        print(f"  Score consistency: {consistency_rating}")
        if metadata.get('rag_examples_used', 0) > 0:
            print(f"  RAG enhancement: {'Improved consistency' if total_stats.get('std_dev', 999) < 3 else 'Standard variance'}")
    
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
        consistency_rating = 'High' if total_stats.get('std_dev', 999) < 2 else 'Medium' if total_stats.get('std_dev', 999) < 5 else 'Low'
        print(f"Consistency Rating: {consistency_rating}")
        
        if metadata.get('rag_examples_used', 0) > 0:
            print(f"RAG Enhancement: Active ({metadata.get('rag_examples_used')} examples used)")
        else:
            print("RAG Enhancement: None (no similar conversations found)")
    
    print(f"\nOverall Feedback:")
    print(f"{results.get('overall_feedback', 'N/A')}")
    print("="*80)

def save_results_to_file(results: Dict[str, Any], filename: str = "rag_enhanced_evaluation_results.json"):
    """Save evaluation results to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def get_database_stats(vector_store: RAGVectorStore) -> Dict[str, Any]:
    """Get statistics about the evaluation database"""
    try:
        conn = sqlite3.connect(vector_store.db_path)
        cursor = conn.cursor()
        
        # Count total evaluations
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        total_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(evaluation_date), MAX(evaluation_date) FROM evaluations WHERE evaluation_date IS NOT NULL")
        date_range = cursor.fetchone()
        
        # Get average scores
        cursor.execute("""
            SELECT AVG(total_score), AVG(clarity_score), AVG(specificity_score), 
                   AVG(context_score), AVG(format_score), AVG(tone_persona_score),
                   AVG(completeness_score), AVG(conciseness_score)
            FROM evaluations WHERE total_score > 0
        """)
        avg_scores = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_evaluations": total_count,
            "date_range": date_range,
            "average_scores": {
                "total": round(avg_scores[0], 2) if avg_scores[0] else 0,
                "clarity": round(avg_scores[1], 2) if avg_scores[1] else 0,
                "specificity": round(avg_scores[2], 2) if avg_scores[2] else 0,
                "context": round(avg_scores[3], 2) if avg_scores[3] else 0,
                "format": round(avg_scores[4], 2) if avg_scores[4] else 0,
                "tone_persona": round(avg_scores[5], 2) if avg_scores[5] else 0,
                "completeness": round(avg_scores[6], 2) if avg_scores[6] else 0,
                "conciseness": round(avg_scores[7], 2) if avg_scores[7] else 0,
            }
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {"total_evaluations": 0, "date_range": None, "average_scores": {}}

def main():
    """Main function to run the RAG-enhanced evaluation"""
    print("Starting RAG-Enhanced Deterministic Prompt Quality Evaluation...")
    print(f"Configuration: {EVALUATION_CONFIG['num_runs']} runs, temp={EVALUATION_CONFIG['temperature']}")
    print(f"RAG Settings: max_examples={EVALUATION_CONFIG['max_examples']}, similarity_threshold={EVALUATION_CONFIG['similarity_threshold']}")
    
    # Load conversation from conversation.json file
    conversation_file = "conversation2.json"
    
    try:
        print(f"Loading conversation from {conversation_file}...")
        conversation_data = load_conversation_from_file(conversation_file)
        print(f"Loaded conversation with {len(conversation_data)} messages")
        
        # Initialize vector store and show database stats
        if EVALUATION_CONFIG["rag_enabled"]:
            print("\nInitializing RAG vector store...")
            vector_store = RAGVectorStore()
            db_stats = get_database_stats(vector_store)
            print(f"Database contains {db_stats['total_evaluations']} previous evaluations")
            if db_stats['date_range'][0]:
                print(f"Evaluation history: {db_stats['date_range'][0]} to {db_stats['date_range'][1]}")
            if db_stats['total_evaluations'] > 0:
                avg_scores = db_stats['average_scores']
                print(f"Historical average total score: {avg_scores['total']}/100")
        
        # Run RAG-enhanced evaluation
        print("\nRunning RAG-enhanced evaluation for improved consistency...")
        results = evaluate_conversation_with_rag(conversation_data)
        
        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_results_to_file(results)
        
        print("\nRAG-enhanced evaluation completed successfully!")
        
        # Print improvement summary
        metadata = results.get("evaluation_metadata", {})
        if metadata.get("rag_examples_used", 0) > 0:
            total_stats = metadata.get("total_stats", {})
            std_dev = total_stats.get("std_dev", 0)
            print(f"\nRAG Enhancement Summary:")
            print(f"  Examples used: {metadata.get('rag_examples_used')}")
            print(f"  Score variance: {std_dev}")
            print(f"  Consistency: {'Excellent' if std_dev < 1 else 'Good' if std_dev < 2 else 'Fair'}")
        else:
            print(f"\nNo similar conversations found for RAG enhancement.")
            print(f"This evaluation will be stored for future RAG improvements.")
        
    except FileNotFoundError:
        print(f"Error: {conversation_file} not found in the current directory.")
        print("Please make sure conversation.json exists in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

# To use this RAG-enhanced script:
# 1. Install: pip install google-generativeai numpy scikit-learn
# 2. Make sure conversation1.json exists in the same directory
# 3. Run: python rag_enhanced_evaluator.py
#
# Key RAG enhancements:
# - SQLite database for storing past evaluations
# - TF-IDF vectorization for conversation similarity
# - Cosine similarity for retrieving relevant examples
# - Dynamic in-context examples for consistent scoring
# - Automatic storage of new evaluations for future RAG
# - Statistical analysis of RAG impact on consistency
# - Configurable similarity thresholds and example limits