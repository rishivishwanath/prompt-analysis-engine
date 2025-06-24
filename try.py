import json
import sqlite3
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import google.generativeai as genai
from datetime import datetime

# Configuration
DIMENSIONS = ["clarity", "specificity", "context", "format", "tone_persona", "completeness", "conciseness"]
INTERACTION_DIMS = ["iteration_quality", "adaptation", "follow_up_effectiveness"]

CONFIG = {
    "num_runs": 3,
    "temperature": 0.4,
    "top_p": 0.1,
    "top_k": 1,
    "max_examples": 3,
    "similarity_threshold": 0.85,
    "reuse_threshold": 0.98,  # New: if similarity > this, return previous result
    "db_path": "evaluation_history.db",
    "vectorizer_cache": "vectorizer_cache.pkl"
}

class RAGVectorStore:
    def __init__(self):
        self.db_path = CONFIG["db_path"]
        self.vectorizer_cache_path = CONFIG["vectorizer_cache"]
        self.vectorizer = None
        self.conversation_vectors = None
        self.conversation_ids = []
        self._init_database()
        self._load_vectorizer()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                conversation_text TEXT NOT NULL,
                conversation_hash TEXT NOT NULL,
                results_json TEXT NOT NULL,
                evaluation_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_vectorizer(self):
        try:
            with open(self.vectorizer_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.vectorizer = cache_data['vectorizer']
                self.conversation_vectors = cache_data['vectors']
                self.conversation_ids = cache_data['conversation_ids']
        except:
            self._rebuild_vectorizer()
    
    def _rebuild_vectorizer(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT conversation_id, conversation_text FROM evaluations")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            self.conversation_vectors = None
            self.conversation_ids = []
            return
        
        self.conversation_ids = [row[0] for row in rows]
        conversation_texts = [row[1] for row in rows]
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.conversation_vectors = self.vectorizer.fit_transform(conversation_texts)
        self._save_vectorizer()
    
    def _save_vectorizer(self):
        cache_data = {
            'vectorizer': self.vectorizer,
            'vectors': self.conversation_vectors,
            'conversation_ids': self.conversation_ids
        }
        with open(self.vectorizer_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def check_for_reuse(self, conversation_text: str):
        """Check if we should reuse a previous result based on high similarity"""
        if not self.conversation_ids or self.conversation_vectors is None:
            return None
        
        query_vector = self.vectorizer.transform([conversation_text])
        similarities = cosine_similarity(query_vector, self.conversation_vectors).flatten()
        max_similarity = np.max(similarities)
        
        if max_similarity > CONFIG["reuse_threshold"]:
            best_match_idx = np.argmax(similarities)
            conversation_id = self.conversation_ids[best_match_idx]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT results_json FROM evaluations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                results = json.loads(row[0])
                results["reused_from_similarity"] = float(max_similarity)
                return results
        
        return None
    
    def get_similar_conversations(self, conversation_text: str):
        """Get similar conversations for RAG examples"""
        if not self.conversation_ids or self.conversation_vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([conversation_text])
        similarities = cosine_similarity(query_vector, self.conversation_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:CONFIG["max_examples"]]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        examples = []
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < CONFIG["similarity_threshold"]:
                continue
            
            conversation_id = self.conversation_ids[idx]
            cursor.execute("SELECT conversation_text, results_json FROM evaluations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            
            if row:
                results = json.loads(row[1])
                examples.append({
                    'text': row[0][:300] + "...",
                    'scores': {dim: results.get(dim, {}).get('score', 0) for dim in DIMENSIONS + INTERACTION_DIMS},
                    'similarity': float(similarity)
                })
        
        conn.close()
        return examples
    
    def store_evaluation(self, conversation_text: str, results: dict):
        """Store new evaluation result"""
        conversation_hash = hashlib.sha256(conversation_text.encode()).hexdigest()
        conversation_id = conversation_hash[:16]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO evaluations 
            (conversation_id, conversation_text, conversation_hash, results_json, evaluation_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, conversation_text, conversation_hash, json.dumps(results), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        # Update vectorizer
        if conversation_id not in self.conversation_ids:
            self.conversation_ids.append(conversation_id)
            if self.conversation_vectors is None:
                self.conversation_vectors = self.vectorizer.fit_transform([conversation_text])
            else:
                new_vector = self.vectorizer.transform([conversation_text])
                self.conversation_vectors = np.vstack([self.conversation_vectors.toarray(), new_vector.toarray()])
                from scipy.sparse import csr_matrix
                self.conversation_vectors = csr_matrix(self.conversation_vectors)
            self._save_vectorizer()

def initialize_gemini():
    genai.configure(api_key="AIzaSyB5LijGTV0692DQauzD1da9v3qY1B5wggQ")
    return genai.GenerativeModel('gemini-1.5-flash')

def extract_human_prompts(conversation: list) -> str:
    """Extract only human messages from the conversation"""
    human_messages = []
    for i, message in enumerate(conversation, 1):
        if message["sender_type"] == "human":
            human_messages.append(f"Prompt {i}: {message['message']}")
    return "\n".join(human_messages)

def create_evaluation_prompt(human_prompts_text: str, examples: list = None) -> str:
    examples_section = ""
    if examples:
        examples_section = "\n\nPREVIOUS EXAMPLES (for consistency):\n"
        for i, ex in enumerate(examples, 1):
            examples_section += f"Example {i} (similarity: {ex['similarity']:.2f}):\n"
            examples_section += f"Human Prompts: {ex['text']}\nScores: {ex['scores']}\n\n"
        examples_section += "Use these for scoring consistency.\n" + "-"*50 + "\n"
    
    return f"""You are evaluating the prompting skills of a HUMAN USER. Focus ONLY on the human's prompts/messages, NOT the AI assistant's responses. {examples_section}

HUMAN PROMPTS TO EVALUATE:
{human_prompts_text}

Evaluate ONLY the human's prompting ability based on their messages. Score each dimension 1-10:

1. Clarity: Are the human's messages clear and unambiguous?
2. Specificity: Does the human clearly define their task/goal/problem?
3. Context: Does the human provide sufficient background information?
4. Format: Does the human specify desired output format when relevant?
5. Tone/Persona: Does the human communicate with appropriate tone?
6. Completeness: Does the human provide all necessary information?
7. Conciseness: Are the human's messages concise without missing important info?
8. Iteration Quality: Does the human refine their requests effectively across the conversation?
9. Adaptation: Does the human adjust their approach when needed?
10. Follow-up Effectiveness: Does the human ask good clarifying questions or provide helpful follow-ups?

Focus on evaluating the HUMAN'S communication and prompting skills, not the AI's responses.

Return valid JSON:
{{
  "clarity": {{"score": X, "reason": "explanation focusing on human's clarity"}},
  "specificity": {{"score": X, "reason": "explanation focusing on human's specificity"}},
  "context": {{"score": X, "reason": "explanation focusing on human's context provision"}},
  "format": {{"score": X, "reason": "explanation focusing on human's format specification"}},
  "tone_persona": {{"score": X, "reason": "explanation focusing on human's tone"}},
  "completeness": {{"score": X, "reason": "explanation focusing on human's completeness"}},
  "conciseness": {{"score": X, "reason": "explanation focusing on human's conciseness"}},
  "iteration_quality": {{"score": X, "reason": "explanation focusing on human's iteration skills"}},
  "adaptation": {{"score": X, "reason": "explanation focusing on human's adaptation"}},
  "follow_up_effectiveness": {{"score": X, "reason": "explanation focusing on human's follow-up skills"}},
  "total_score": X,
  "percentage_score": "XX%",
  "overall_feedback": "comprehensive feedback on the human's prompting skills"
}}"""

def parse_response(response_text: str) -> dict:
    # Clean JSON from response
    text = response_text.strip()
    if '```json' in text:
        start = text.find('```json') + 7
        end = text.find('```', start)
        text = text[start:end].strip()
    
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    return json.loads(text)

def run_evaluation(model, human_prompts_text: str, examples: list) -> dict:
    prompt = create_evaluation_prompt(human_prompts_text, examples)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            top_k=CONFIG["top_k"],
            max_output_tokens=2048
        )
    )
    
    return parse_response(response.text)

def aggregate_results(results: list) -> dict:
    if not results:
        return {"error": "No valid results"}
    
    # Use median scores for stability
    aggregated = {}
    all_dims = DIMENSIONS + INTERACTION_DIMS
    
    for dim in all_dims:
        scores = [r[dim]["score"] for r in results if dim in r]
        if scores:
            aggregated[dim] = {
                "score": np.median(scores),
                "reason": results[0][dim]["reason"]
            }
    
    # Calculate totals
    total_scores = [sum(r[dim]["score"] for dim in all_dims if dim in r) for r in results]
    final_total = np.median(total_scores) if total_scores else 0
    
    aggregated.update({
        "total_score": final_total,
        "percentage_score": f"{(final_total/100)*100:.0f}%",
        "overall_feedback": results[0].get("overall_feedback", ""),
        "runs_completed": len(results)
    })
    
    return aggregated

def evaluate_conversation(conversation_data: list) -> dict:
    print("Initializing evaluation...")
    print(f"Total messages in conversation: {len(conversation_data)}")
    
    # Extract only human prompts
    human_prompts_text = extract_human_prompts(conversation_data)
    human_message_count = len([msg for msg in conversation_data if msg["sender_type"] == "human"])
    print(f"Human messages found: {human_message_count}")
    
    if not human_prompts_text.strip():
        return {"error": "No human messages found in conversation"}
    
    model = initialize_gemini()
    vector_store = RAGVectorStore()
    
    # Check if we should reuse previous result
    print("Checking for high-similarity matches...")
    reused_result = vector_store.check_for_reuse(human_prompts_text)
    if reused_result:
        print(f"Reusing previous result (similarity: {reused_result['reused_from_similarity']:.3f})")
        return reused_result
    
    # Get RAG examples
    print("Getting similar conversations for RAG...")
    examples = vector_store.get_similar_conversations(human_prompts_text)
    print(f"Found {len(examples)} similar conversations")
    
    # Run multiple evaluations
    print(f"Running {CONFIG['num_runs']} evaluations...")
    results = []
    for i in range(CONFIG['num_runs']):
        result = run_evaluation(model, human_prompts_text, examples)
        results.append(result)
        print(f"Completed run {i+1}")
    
    # Aggregate results
    final_results = aggregate_results(results)
    
    # Store for future use
    vector_store.store_evaluation(human_prompts_text, final_results)
    print("Results stored for future RAG")
    
    return final_results

def print_results(results: dict):
    print("\n" + "="*60)
    print("HUMAN PROMPTING SKILLS EVALUATION")
    print("="*60)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    if "reused_from_similarity" in results:
        print(f"REUSED RESULT (Similarity: {results['reused_from_similarity']:.3f})")
        print("-"*60)
    
    # Dimension scores
    print("Core Prompting Dimensions:")
    for dim in DIMENSIONS:
        if dim in results:
            label = dim.replace("_", " ").title()
            print(f"  {label}: {results[dim]['score']}/10")
    
    print("\nInteraction Quality:")
    for dim in INTERACTION_DIMS:
        if dim in results:
            label = dim.replace("_", " ").title()
            print(f"  {label}: {results[dim]['score']}/10")
    
    print(f"\nTotal Score: {results.get('total_score', 0)}/100")
    print(f"Percentage: {results.get('percentage_score', '0%')}")
    print(f"\nOverall Feedback on Human's Prompting Skills:")
    print(f"{results.get('overall_feedback', 'N/A')}")

def main():
    conversation_file = "conversation.json"
    
    with open(conversation_file, 'r') as f:
        data = json.load(f)
    
    conversation_data = data.get('conversation', [])
    print(f"Loaded {len(conversation_data)} messages")
    
    results = evaluate_conversation(conversation_data)
    print_results(results)
    
    # Save results
    with open("human_prompt_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to human_prompt_evaluation_results.json")

if __name__ == "__main__":
    main()