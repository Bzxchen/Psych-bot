"""
Question Type / Intent Classifier for PsychBot.

This module classifies questions into different types to enable
tailored responses and appropriate safety measures.
"""

from typing import Dict, List, Optional
from enum import Enum


class QuestionType(Enum):
    """Types of questions the bot can classify."""
    CONCEPTUAL = "conceptual"  # Definitions, theories, "what is X?"
    APPLIED = "applied"  # Examples, real-life use, "how does X work in practice?"
    EMOTIONAL = "emotional"  # Distress, personal feelings, seeking comfort
    CLARIFICATION = "clarification"  # Follow-up questions, "can you explain more?"
    CRISIS = "crisis"  # Urgent mental health concerns requiring professional help


class QuestionClassifier:
    """
    Classifies user questions by intent/type.
    
    Currently uses rule-based classification. Can be extended with
    ML models (Logistic Regression, SVM, or fine-tuned BERT) in the future.
    """
    
    def __init__(self):
        """Initialize the classifier with keyword patterns."""
        # Crisis indicators - highest priority
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "want to die",
            "hurting myself", "self harm", "cutting", "overdose",
            "emergency", "urgent help", "immediate", "right now",
            "can't cope", "can't go on", "hopeless", "no way out"
        ]
        
        # Emotional indicators
        self.emotional_keywords = [
            "feel", "feeling", "emotions", "anxious", "worried",
            "scared", "afraid", "sad", "depressed", "lonely",
            "overwhelmed", "stressed", "panic", "fear", "worry",
            "help me", "struggling", "difficult", "hard time",
            "coping", "deal with", "manage", "my situation"
        ]
        
        # Applied indicators
        self.applied_keywords = [
            "example", "examples", "case study", "real life",
            "in practice", "how to", "how do", "how can",
            "apply", "application", "use", "used", "using",
            "work in", "happens when", "scenario", "situation"
        ]
        
        # Conceptual indicators
        self.conceptual_keywords = [
            "what is", "define", "definition", "meaning",
            "explain", "theory", "theories", "concept",
            "understand", "difference between", "compare",
            "types of", "kinds of", "categories"
        ]
        
        # Clarification indicators
        self.clarification_keywords = [
            "can you", "could you", "please explain more",
            "more about", "tell me more", "elaborate",
            "clarify", "what do you mean", "i don't understand",
            "confused", "unclear"
        ]
    
    def classify(self, question: str) -> QuestionType:
        """
        Classify a question into one of the question types.
        
        Priority order:
        1. Crisis (highest priority - safety first)
        2. Emotional
        3. Applied
        4. Conceptual
        5. Clarification (default)
        
        Args:
            question: The user's question
            
        Returns:
            QuestionType enum value
        """
        question_lower = question.lower()
        
        # Check for crisis indicators first (safety priority)
        for keyword in self.crisis_keywords:
            if keyword in question_lower:
                return QuestionType.CRISIS
        
        # Count keyword matches for each category
        emotional_score = sum(1 for kw in self.emotional_keywords if kw in question_lower)
        applied_score = sum(1 for kw in self.applied_keywords if kw in question_lower)
        conceptual_score = sum(1 for kw in self.conceptual_keywords if kw in question_lower)
        clarification_score = sum(1 for kw in self.clarification_keywords if kw in question_lower)
        
        # Return the category with the highest score
        scores = {
            QuestionType.EMOTIONAL: emotional_score,
            QuestionType.APPLIED: applied_score,
            QuestionType.CONCEPTUAL: conceptual_score,
            QuestionType.CLARIFICATION: clarification_score
        }
        
        max_score = max(scores.values())
        
        # If no clear match, default to conceptual for psychology questions
        if max_score == 0:
            return QuestionType.CONCEPTUAL
        
        # Return the type with the highest score
        for qtype, score in scores.items():
            if score == max_score:
                return qtype
    
    def get_classification_metadata(self, question: str) -> Dict:
        """
        Get detailed classification information.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with classification details
        """
        qtype = self.classify(question)
        
        return {
            "type": qtype,
            "type_name": qtype.value,
            "confidence": "high" if qtype != QuestionType.CLARIFICATION else "medium"
        }


class ResponseStructuring:
    """
    Structures responses based on question type and adds appropriate
    safety measures and professional resource redirects.
    """
    
    # Professional resources
    CRISIS_RESOURCES = {
        "National Suicide Prevention Lifeline": "988",
        "Crisis Text Line": "Text HOME to 741741",
        "National Alliance on Mental Illness (NAMI)": "1-800-950-NAMI (6264)",
        "Substance Abuse and Mental Health Services Administration": "1-800-662-HELP (4357)"
    }
    
    EMERGENCY_MESSAGE = """
⚠️ IMPORTANT: If you or someone you know is in immediate danger, please call 911 or go to your nearest emergency room.

For immediate mental health support:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• These services are available 24/7, free, and confidential.
"""
    
    @staticmethod
    def get_safety_disclaimer() -> str:
        """Get a general safety disclaimer."""
        return "\n\n💙 **Important Note**: This bot provides educational information and support, but it is not a replacement for professional mental health care. If you're experiencing a mental health crisis or need immediate support, please contact a mental health professional or crisis hotline."
    
    @staticmethod
    def get_crisis_response() -> str:
        """Get response for crisis situations."""
        response = ResponseStructuring.EMERGENCY_MESSAGE
        response += "\n\nI'm concerned about what you've shared. While I'm here to provide information and support, your situation sounds serious and requires immediate professional attention."
        response += "\n\n**Please reach out to one of these resources right away:**\n"
        for resource, contact in ResponseStructuring.CRISIS_RESOURCES.items():
            response += f"• {resource}: {contact}\n"
        response += "\nThese professionals are trained to help and can provide the support you need. You don't have to go through this alone."
        return response
    
    @staticmethod
    def structure_response(
        base_response: str,
        question_type: QuestionType,
        is_emotional: bool = False
    ) -> str:
        """
        Structure a response based on question type.
        
        Args:
            base_response: The base response from the bot
            question_type: The classified question type
            is_emotional: Whether the question has emotional components
            
        Returns:
            Structured response with appropriate tone and safety measures
        """
        # Handle crisis separately
        if question_type == QuestionType.CRISIS:
            return ResponseStructuring.get_crisis_response()
        
        # Add emotional support for emotional questions
        if question_type == QuestionType.EMOTIONAL or is_emotional:
            emotional_prefix = "I understand this can be difficult to talk about, and I want you to know that your feelings are valid. "
            base_response = emotional_prefix + base_response
        
        # Add examples for applied questions
        if question_type == QuestionType.APPLIED:
            # The base response should already include examples, but we can emphasize
            pass  # Could add example prompts here if needed
        
        # Add safety disclaimer for emotional or applied questions
        if question_type in [QuestionType.EMOTIONAL, QuestionType.APPLIED]:
            base_response += ResponseStructuring.get_safety_disclaimer()
        
        # Add note about question type (for transparency/debugging)
        type_labels = {
            QuestionType.CONCEPTUAL: "📚 Conceptual",
            QuestionType.APPLIED: "💡 Applied",
            QuestionType.EMOTIONAL: "💙 Emotional",
            QuestionType.CLARIFICATION: "❓ Clarification",
            QuestionType.CRISIS: "⚠️ Crisis"
        }
        type_note = f"\n\n*[Detected question type: {type_labels.get(question_type, question_type.value)}]*"
        base_response += type_note
        
        return base_response
    
    @staticmethod
    def get_professional_resources() -> str:
        """Get a list of professional resources for general reference."""
        resources = "\n\n**Professional Resources for Mental Health Support:**\n"
        resources += "• **Therapy/Counseling**: Psychology Today (psychologytoday.com) - Find therapists in your area\n"
        resources += "• **Support Groups**: NAMI (nami.org) - Local support groups and resources\n"
        resources += "• **Online Therapy**: BetterHelp, Talkspace - Licensed therapists online\n"
        resources += "• **Your Primary Care Doctor**: Can provide referrals to mental health specialists\n"
        resources += "• **Local Mental Health Centers**: Check your county or city health department\n"
        return resources
