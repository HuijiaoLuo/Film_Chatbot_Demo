# moviebot_app.py
import streamlit as st
import re
from openai import OpenAI
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import json
from neo4j.graph import Node, Relationship

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
        try:
            self.llm = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
            self.driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
            )
            st.session_state.bot_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize MovieBot: {e}")
            st.stop()
    
    def extract_pure_cypher(self, text):
        """Extract pure Cypher from LLM response"""
        cleaned = re.sub(r'```.*?\n', '', text)
        cleaned = re.sub(r'\n```.*?$', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        return cleaned.strip()
    
    def neo4j_result_to_dict(self, result):
        """Convert Neo4j result to serializable dictionary"""
        serializable_results = []
        for record in result:
            record_dict = {}
            for key, value in record.items():
                if isinstance(value, Node):
                    # Convert Node to dictionary
                    record_dict[key] = dict(value.items())
                    record_dict[key]['_labels'] = list(value.labels)
                    record_dict[key]['_id'] = value.id
                elif isinstance(value, Relationship):
                    # Convert Relationship to dictionary
                    record_dict[key] = dict(value.items())
                    record_dict[key]['_type'] = value.type
                    record_dict[key]['_start_node'] = value.start_node.id
                    record_dict[key]['_end_node'] = value.end_node.id
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    # Handle lists and other iterables
                    record_dict[key] = [self._convert_item(item) for item in value]
                else:
                    record_dict[key] = self._convert_item(value)
            serializable_results.append(record_dict)
        return serializable_results
    
    def _convert_item(self, item):
        """Convert individual items to serializable format"""
        if isinstance(item, (Node, Relationship)):
            return str(item)  # Fallback to string representation
        elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            return [self._convert_item(subitem) for subitem in item]
        else:
            return item
    
    def query_neo4j(self, cypher_query):
        """Execute Cypher query and return serializable results"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                # Convert to serializable format
                return self.neo4j_result_to_dict(result)
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
        # 首先简化结果，只保留关键信息
        simplified_results = []
        for result in results[:3]:  # 只取前3个结果
            simplified = {}
            for key, value in result.items():
                if isinstance(value, dict) and '_labels' in value:
                    # 这是Node对象
                    if 'Movie' in value['_labels']:
                        simplified[key] = {
                            'type': 'Movie',
                            'title': value.get('title', 'Unknown Title'),
                            'year': value.get('year', 'Unknown Year')
                        }
                    elif 'Person' in value['_labels']:
                        simplified[key] = {
                            'type': 'Person', 
                            'name': value.get('name', 'Unknown Name')
                        }
                else:
                    simplified[key] = value
            simplified_results.append(simplified)
        
        prompt = f"""
        You are MovieBot, a friendly movie expert assistant. 
        
        User asked: "{user_query}"
        Database results: {json.dumps(simplified_results, ensure_ascii=False)}
        
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
            # 如果LLM调用失败，返回简化响应
            if results:
                return f"I found {len(results)} results! 🎬 What would you like to know next?"
            else:
                return "I couldn't find any results for that query. Would you like to ask something else? 🎥"

# Streamlit app configuration
st.set_page_config(
    page_title="🎬 MovieBot - Your Personal Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize bot
if "bot" not in st.session_state:
    try:
        st.session_state.bot = MovieBot()
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
                raw_cypher = st.session_state.bot.generate_cypher(prompt)
                if raw_cypher:
                    cypher_query = st.session_state.bot.extract_pure_cypher(raw_cypher)
                    st.write(f"*Executing query:* `{cypher_query}`")
                    
                    # Execute query
                    results = st.session_state.bot.query_neo4j(cypher_query)
                    
                    # Generate friendly response
                    response = st.session_state.bot.generate_friendly_response(prompt, results)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
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