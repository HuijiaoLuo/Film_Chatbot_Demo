# moviebot_app.py
import streamlit as st
import re
from openai import OpenAI
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import json
# Load environment variables
load_dotenv()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "bot_initialized" not in st.session_state:
    st.session_state.bot_initialized = False

class MovieBot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        st.session_state.bot_initialized = True
    
    def extract_pure_cypher(self, text):
        """Extract pure Cypher from LLM response"""
        cleaned = re.sub(r'```.*?\n', '', text)
        cleaned = re.sub(r'\n```.*?$', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        return cleaned.strip()
    
    def query_neo4j(self, cypher_query):
        """Execute Cypher query"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Database query failed: {e}")
            return []
    
    def generate_cypher(self, user_query):
        prompt = f"""
        Generate a Neo4j Cypher query based on the user question.

        User question: {user_query}

        Database schema:
        - Nodes: Movie, Person, Actor, Director
        - Relationships: ACTED_IN, DIRECTED
        - Properties:
          * Movie: title, year, plot, released, budget, languages, countries, imdbRating
          * Person: name, born, bornIn, bio
          * ACTED_IN: role

        Return ONLY the Cypher query statement, nothing else.
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM query generation failed: {e}")
            return None
    
    def generate_friendly_response(self, user_query, results):
        """Generate friendly response using LLM"""
        prompt = f"""
        You are MovieBot, a friendly movie expert assistant. 
        
        User asked: "{user_query}"
        Database results: {json.dumps(results[:3]) if results else "No results found"}
        
        Respond in a warm, engaging way:
        - Use emojis and friendly tone
        - Summarize the results naturally
        - Offer recommendations if relevant
        - Ask follow-up questions
        - Keep it concise but informative
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

# Streamlit app configuration
st.set_page_config(
    page_title="🎬 MovieBot - Your Personal Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize bot
if not st.session_state.bot_initialized:
    try:
        bot = MovieBot()
    except Exception as e:
        st.error(f"Failed to initialize MovieBot: {e}")
        st.stop()

# Sidebar
with st.sidebar:
    st.title("🎬 MovieBot")
    st.markdown("Your personal movie expert and recommendation assistant!")
    
    st.divider()
    st.subheader("💬 Chat Controls")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("👤 Your Profile")
    if st.session_state.user_profile:
        st.json(st.session_state.user_profile)
    else:
        st.info("Your preferences will appear here as we chat!")
    
    st.divider()
    st.subheader("❓ How to use")
    st.markdown("""
    - Ask about movies: "best sci-fi movies"
    - Get recommendations: "movies like Inception"
    - Actor information: "Tom Hanks movies"
    - Movie details: "tell me about The Godfather"
    """)

# Main content area
st.title("💬 MovieBot Chat")
st.caption("Ask me anything about movies and get personalized recommendations!")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about movies?"):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🎬 Thinking about movies..."):
            try:
                # Generate Cypher query
                raw_cypher = bot.generate_cypher(prompt)
                if raw_cypher:
                    cypher_query = bot.extract_pure_cypher(raw_cypher)
                    
                    # Execute query
                    results = bot.query_neo4j(cypher_query)
                    
                    # Generate friendly response
                    response = bot.generate_friendly_response(prompt, results)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Update user profile (simple version)
                    if "like" in prompt.lower() or "love" in prompt.lower():
                        if "sci-fi" in prompt.lower():
                            st.session_state.user_profile["favorite_genre"] = "sci-fi"
                        elif "comedy" in prompt.lower():
                            st.session_state.user_profile["favorite_genre"] = "comedy"
                
                else:
                    error_msg = "Sorry, I couldn't generate a query for that. Could you rephrase?"
                    st.markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Quick action buttons
st.divider()
st.subheader("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🎭 Top Rated Movies", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "Show me top rated movies"})
        st.rerun()

with col2:
    if st.button("🌟 Popular Actors", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "Who are the most popular actors?"})
        st.rerun()

with col3:
    if st.button("🎬 Recent Releases", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "What are the latest movies?"})
        st.rerun()