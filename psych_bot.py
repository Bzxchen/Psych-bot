import os
import sys
import pickle
import hashlib
from typing import List, Optional, Tuple
import numpy as np
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
from question_classifier import QuestionClassifier, QuestionType, ResponseStructuring

# Load environment variables
load_dotenv()


class PsychBot:
    """
    A comforting, informative psychology bot that can answer questions
    based on a PDF textbook using vector embeddings and semantic search.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PsychBot.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass it as a parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.textbook_content = ""
        self.model = "gpt-4"  # Can be changed to gpt-3.5-turbo for cost savings
        self.embedding_model = "text-embedding-3-small"  # Efficient and cost-effective
        self.chunks = []  # List of text chunks
        self.embeddings = []  # List of embedding vectors
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        self.top_k = 5  # Number of relevant chunks to retrieve
        self.classifier = QuestionClassifier()  # Question type classifier
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at a sentence boundary if possible
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # Only break if we're past 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + len(chunk)
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error creating embedding: {str(e)}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_relevant_chunks(self, question: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find the most relevant chunks for a question using semantic search.
        
        Args:
            question: The user's question
            top_k: Number of chunks to return (defaults to self.top_k)
            
        Returns:
            List of tuples (chunk_text, similarity_score) sorted by relevance
        """
        if not self.embeddings:
            return []
        
        top_k = top_k or self.top_k
        
        # Get embedding for the question
        question_embedding = self._get_embedding(question)
        
        # Calculate similarity scores
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(question_embedding, chunk_embedding)
            similarities.append((self.chunks[i], similarity))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _get_cache_path(self, pdf_path: str) -> str:
        """
        Generate cache file path based on PDF path and chunk parameters.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the cache file
        """
        # Create a hash based on PDF path and chunk parameters
        cache_key = f"{pdf_path}_{self.chunk_size}_{self.chunk_overlap}_{self.embedding_model}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return f"embeddings_cache_{cache_hash}.pkl"
    
    def ingest_pdf(self, pdf_path: str, use_cache: bool = True, append: bool = False) -> str:
        """
        Extract text from a PDF file and create embeddings.
        
        Args:
            pdf_path: Path to the PDF file
            use_cache: Whether to use cached embeddings if available
            append: If True, append to existing chunks/embeddings instead of replacing
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        cache_path = self._get_cache_path(pdf_path)
        
        # Try to load from cache (only if not appending)
        if use_cache and not append and os.path.exists(cache_path):
            print(f"Loading embeddings from cache: {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if append:
                        self.chunks.extend(cached_data['chunks'])
                        self.embeddings.extend(cached_data['embeddings'])
                        self.textbook_content += "\n\n" + cached_data['textbook_content']
                    else:
                        self.chunks = cached_data['chunks']
                        self.embeddings = cached_data['embeddings']
                        self.textbook_content = cached_data['textbook_content']
                print(f"✓ Loaded {len(cached_data['chunks'])} chunks from cache.")
                return self.textbook_content
            except Exception as e:
                print(f"Warning: Could not load cache ({e}). Recreating embeddings...")
        
        # Extract text from PDF
        print(f"Reading PDF: {pdf_path}...")
        text_content = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                    if page_num % 10 == 0:
                        print(f"Processed {page_num}/{total_pages} pages...")
                
                pdf_text = "\n\n".join(text_content)
                print(f"Successfully extracted text from {total_pages} pages.")
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        # Chunk the text
        print("Chunking text...")
        new_chunks = self._chunk_text(pdf_text)
        print(f"Created {len(new_chunks)} chunks.")
        
        # Create embeddings
        print("Creating embeddings (this may take a while for large PDFs)...")
        new_embeddings = []
        for i, chunk in enumerate(new_chunks):
            if (i + 1) % 50 == 0:
                print(f"Embedded {i + 1}/{len(new_chunks)} chunks...")
            embedding = self._get_embedding(chunk)
            new_embeddings.append(embedding)
        
        print(f"✓ Created embeddings for {len(new_embeddings)} chunks.")
        
        # Append or replace
        if append:
            self.chunks.extend(new_chunks)
            self.embeddings.extend(new_embeddings)
            self.textbook_content += "\n\n---\n\n" + pdf_text
        else:
            self.chunks = new_chunks
            self.embeddings = new_embeddings
            self.textbook_content = pdf_text
        
        # Save to cache (individual file cache)
        if use_cache:
            try:
                cache_data = {
                    'chunks': new_chunks,
                    'embeddings': new_embeddings,
                    'textbook_content': pdf_text
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"✓ Cached embeddings to {cache_path}")
            except Exception as e:
                print(f"Warning: Could not save cache ({e})")
        
        return self.textbook_content
    
    def ingest_data_folder(self, data_folder: str = "data", use_cache: bool = True) -> int:
        """
        Ingest all PDF files from a data folder.
        
        Args:
            data_folder: Path to the folder containing PDF files
            use_cache: Whether to use cached embeddings if available
            
        Returns:
            Number of PDFs successfully ingested
        """
        if not os.path.exists(data_folder):
            print(f"Data folder '{data_folder}' not found.")
            return 0
        
        if not os.path.isdir(data_folder):
            print(f"'{data_folder}' is not a directory.")
            return 0
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in '{data_folder}'.")
            return 0
        
        pdf_files.sort()  # Process in alphabetical order
        print(f"Found {len(pdf_files)} PDF file(s) in '{data_folder}':")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        print()
        
        # Reset chunks and embeddings for fresh start
        self.chunks = []
        self.embeddings = []
        self.textbook_content = ""
        
        # Ingest each PDF
        successful = 0
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(data_folder, pdf_file)
            try:
                print(f"[{i+1}/{len(pdf_files)}] Processing: {pdf_file}")
                self.ingest_pdf(pdf_path, use_cache=use_cache, append=(i > 0))
                successful += 1
                print(f"✓ Successfully ingested {pdf_file}\n")
            except Exception as e:
                print(f"✗ Error processing {pdf_file}: {e}\n")
        
        print(f"✓ Ingested {successful}/{len(pdf_files)} PDF file(s).")
        print(f"Total chunks: {len(self.chunks)}")
        return successful
    
    def _create_system_prompt(self, question_type: QuestionType) -> str:
        """
        Create the system prompt that defines the bot's psychologist-like personality.
        
        Args:
            question_type: The type of question being asked
        """
        base_prompt = """You are a warm, empathetic, and knowledgeable psychology expert. Your role is to help people understand psychological concepts and answer their questions in a comforting, supportive manner.

Key aspects of your communication style:
- Be warm, empathetic, and non-judgmental
- Use clear, accessible language (avoid overly technical jargon when possible)
- Acknowledge the person's feelings and concerns
- Provide informative, evidence-based answers
- Be encouraging and supportive
- When appropriate, validate their experiences
- Break down complex concepts into understandable parts

You have access to psychology resources and textbooks that you can reference to provide accurate, informative answers. Always ground your responses in the resource content when relevant, but also bring warmth and understanding to your explanations."""
        
        # Add type-specific guidance
        if question_type == QuestionType.EMOTIONAL:
            base_prompt += "\n\nIMPORTANT: This question appears to involve emotional distress. Be especially warm, validating, and supportive. Acknowledge their feelings first before providing information."
        elif question_type == QuestionType.APPLIED:
            base_prompt += "\n\nIMPORTANT: This question is asking for practical application. Provide concrete examples, real-world scenarios, and actionable information when possible."
        elif question_type == QuestionType.CONCEPTUAL:
            base_prompt += "\n\nIMPORTANT: This is a conceptual question. Focus on clear definitions, explanations, and theoretical foundations from the resources."
        elif question_type == QuestionType.CLARIFICATION:
            base_prompt += "\n\nIMPORTANT: This is a follow-up or clarification question. Be thorough and clear, building on previous context if available."
        
        return base_prompt
    
    def _create_user_prompt(
        self, 
        question: str, 
        relevant_chunks: List[Tuple[str, float]],
        question_type: QuestionType
    ) -> str:
        """
        Create the user prompt with the question and relevant textbook context.
        
        Args:
            question: The user's question
            relevant_chunks: List of (chunk_text, similarity_score) tuples
            question_type: The type of question being asked
        """
        if not relevant_chunks:
            return question
        
        # Combine relevant chunks
        context_parts = []
        for chunk, score in relevant_chunks:
            context_parts.append(chunk)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Limit total context to avoid token limits (approximately 12000 characters)
        if len(context) > 12000:
            context = context[:12000] + "..."
        
        # Adjust prompt based on question type
        if question_type == QuestionType.APPLIED:
            instruction = "Please provide practical examples, real-world applications, and actionable information based on the resource content."
        elif question_type == QuestionType.EMOTIONAL:
            instruction = "Please be especially warm and validating. Acknowledge their feelings first, then provide supportive information from the resources."
        elif question_type == QuestionType.CONCEPTUAL:
            instruction = "Please provide clear definitions, explanations, and theoretical foundations from the resources."
        else:
            instruction = "Please provide a thoughtful, empathetic response."
        
        prompt = f"""Based on the following relevant excerpts from psychology resources, please answer the user's question in a warm, comforting, and informative manner.

RELEVANT RESOURCE EXCERPTS:
{context}

USER QUESTION: {question}

{instruction}

Your response should:
1. Directly address their question
2. Reference relevant information from the resources when applicable
3. Maintain a warm, supportive, and professional psychologist-like tone
4. Be clear and accessible"""
        
        return prompt
    
    def answer_question(self, question: str, use_textbook: bool = True) -> str:
        """
        Answer a question using semantic search and OpenAI API.
        Classifies the question type and structures the response accordingly.
        
        Args:
            question: The user's question
            use_textbook: Whether to use the ingested textbook content
            
        Returns:
            The bot's response
        """
        if not question.strip():
            return "I'm here to help! Please feel free to ask me any questions about psychology."
        
        # Classify the question
        question_type = self.classifier.classify(question)
        classification_meta = self.classifier.get_classification_metadata(question)
        
        # Handle crisis situations immediately - don't use API, return safety resources
        if question_type == QuestionType.CRISIS:
            return ResponseStructuring.get_crisis_response()
        
        if use_textbook and not self.chunks:
            return ("I haven't ingested a textbook yet. Please use the ingest_pdf() method first, "
                   "or set use_textbook=False to answer without textbook context.")
        
        try:
            # Create system prompt based on question type
            system_prompt = self._create_system_prompt(question_type)
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if use_textbook:
                # Find relevant chunks using semantic search
                relevant_chunks = self._find_relevant_chunks(question)
                user_content = self._create_user_prompt(question, relevant_chunks, question_type)
            else:
                user_content = question
            
            messages.append({"role": "user", "content": user_content})
            
            print("Thinking...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,  # Slightly creative but still focused
                max_tokens=1000
            )
            
            base_response = response.choices[0].message.content
            
            # Structure the response based on question type
            is_emotional = question_type == QuestionType.EMOTIONAL
            structured_response = ResponseStructuring.structure_response(
                base_response, 
                question_type, 
                is_emotional
            )
            
            return structured_response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."


def main():
    """Main CLI interface for the PsychBot."""
    print("=" * 60)
    print("Welcome to PsychBot - Your Comforting Psychology Assistant")
    print("=" * 60)
    print()
    
    # Initialize bot
    try:
        bot = PsychBot()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease create a .env file with your OPENAI_API_KEY, or set it as an environment variable.")
        sys.exit(1)
    
    # Automatically load all PDFs from data folder
    data_folder = "data"
    pdfs_loaded = bot.ingest_data_folder(data_folder)
    
    if pdfs_loaded > 0:
        print(f"✓ {pdfs_loaded} resource(s) loaded and indexed successfully!\n")
    else:
        print("Note: No PDF files found in the 'data' folder.")
        print("You can still ask questions, but I won't have textbook context.\n")
        print("To add resources: Place PDF files in a 'data' folder in the project directory.\n")
    
    # Interactive Q&A loop
    print("You can now ask me questions about psychology!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\nThank you for chatting! Take care of yourself. 💙")
            break
        
        if not question:
            continue
        
        response = bot.answer_question(question)
        print(f"\nPsychBot: {response}\n")


if __name__ == "__main__":
    main()
