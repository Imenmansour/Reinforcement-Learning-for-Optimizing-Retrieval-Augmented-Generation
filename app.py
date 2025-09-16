import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import sqlite3
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from typing import List, Dict, Any
import streamlit as st

# Initialize agent and other variables
# Set up Google Gemini API key
GOOGLE_API_KEY = "AIzaSyC5TYmwohkkEXsTc77XypXRGxP77U_5jH4"  # Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit page configuration
st.set_page_config(page_title="RAG with PPO", page_icon="🤖")
st.title("RAG Pipeline with Feedback and PPO")

# Initialize chat history in session_state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize feedback state
if "feedback_pending" not in st.session_state:
    st.session_state.feedback_pending = None

# Initialize query count
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# Text splitter for chunking files
custom_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    separators=["\n"]
)


def chunk_text_files(directory: str) -> List[Document]:
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = custom_text_splitter.split_text(text)
                for chunk in chunks:
                    document = Document(
                        page_content=chunk,
                        metadata={"source": filename, "frequency": 0}  # Initial frequency
                    )
                    all_documents.append(document)
    return all_documents


# Load documents and create FAISS vector store
@st.cache_resource()
def load_vector_store() -> FAISS:
    data_directory = "/content/data_txt"  # Replace with your data directory
    documents = chunk_text_files(data_directory)
    return FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# PPO Memory Class
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# Actor and Critic Netwo
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='/content/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.pth')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        print(f"State shape before passing to actor network: {state.shape}")

        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='/content/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.pth')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# PPO Agent Class
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, n_epochs=10, epsilon=0.1):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon  # epsilon for exploration-exploitation trade-off

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
        print(f"Memory stored: State={state}, Action={action}, Reward={reward}, Done={done}")
        print(f"Current memory length: {len(self.memory.states)}") 

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state_embedding, doc_frequency=None):
        # Ensure doc_frequency is initialized
        if doc_frequency is None:
            doc_frequency = {}

        # Ensure state_embedding is a tensor and on the correct device
        if not isinstance(state_embedding, T.Tensor):
            state_embedding = T.tensor(state_embedding, dtype=T.float32).to(self.actor.device)
        
        if state_embedding.ndim == 1:
            state_embedding = state_embedding.unsqueeze(0)  # Add batch dimension

        # Get the distribution and value from the actor and critic models
        state = state_embedding.unsqueeze(0)  # Ensure the batch dimension
        dist = self.actor(state)
        value = self.critic(state)

        # Exploration vs Exploitation (epsilon-greedy strategy)
        if np.random.rand() < self.epsilon:
            # Exploration: randomly select an action
            action = np.random.choice(len(dist.probs))
        else:
            # Exploitation: sample an action based on the policy distribution
            action = dist.sample().item()

        # Apply frequency penalty: Encourage exploration of less frequent documents
        if doc_frequency.get(action, 0) < 5:
            # If the document is rare (frequency < 5), allow action
            adjusted_action = action
        else:
            # If the document is frequent (frequency >= 5), encourage exploration
            adjusted_action = np.random.choice(len(dist.probs))  # Random choice as penalty

        # Ensure the action is a tensor and on the correct device
        action_tensor = T.tensor([adjusted_action], dtype=T.long).to(self.actor.device)

        # Get the log probability of the selected action
        probs = dist.log_prob(action_tensor).item()
        value = T.squeeze(value).item()

        return adjusted_action, probs, value





    def learn(self):
        print(f"Starting learning with {len(self.memory.states)} memory entries")
        # Optionally, you can print the first few entries to verify
        for i in range(min(5, len(self.memory.states))):  # Print first 5 entries or all if less than 5
            print(f"Memory[{i}] - State: {self.memory.states[i]}, Action: {self.memory.actions[i]}, Reward: {self.memory.rewards[i]}")
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                  1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


def encode_query_to_state(query: str) -> np.ndarray:
    """
    Convert the query into a vector state using an embedding model.
    
    Args:
        query (str): The user query to encode.
    
    Returns:
        np.ndarray: The vector representation of the query.
    """
    # Use the embedding model to encode the query into a vector
    query_vector = embedding_model.embed_query(query)  # Correct method call
    
    # Return the query as a numpy array (state for the agent)
    return np.array(query_vector, dtype=np.float32)



def generate_answer(agent, db, query):
    # Retrieve documents based on the query using the PPO agent and document frequency
    retrieved_docs,action,probs, vals  = get_document_action(agent, query, db)

    # Combine the retrieved documents into a prompt for generating an answer
    relevant_passages = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""Answer the following question based only on the provided passage.
    QUESTION: {query}
    PASSAGE: {relevant_passages}
    ANSWER: """

    # Simulating the response generation (can use model like GPT here)
    response = model.generate_content(prompt)
    return response.text, retrieved_docs,action,probs, vals 


def get_document_action(agent, query, db, doc_frequency=None, k=5):
    """
    Get the most relevant documents for a query based on PPO action and document frequency.
    
    Args:
        agent: The PPO agent used to select the documents.
        query: The query to be answered.
        db: The vector store to search from.
        doc_frequency: A dictionary that tracks how often each document has been used.
        k: Number of documents to retrieve (this can be influenced by the action).
    
    Returns:
        List[Document]: A list of the retrieved documents.
    """
    # Initialize doc_frequency if it's None
    if doc_frequency is None:
        doc_frequency = {}

    # Convert query to state
    state = encode_query_to_state(query)
    action, probs, vals = agent.choose_action(state)
    
    # Use the agent to choose an action based on the query state and document frequency
    action, _, _ = agent.choose_action(state, doc_frequency)
    
    # Adjust the number of documents to retrieve based on the action
    k = max(action, 1)  

    # Retrieve the documents using similarity search from the vector store (db)
    documents = db.similarity_search(query, k=k)
    
    # Increment frequency for retrieved documents
    for doc in documents:
        if doc.metadata["source"] not in doc_frequency:
            doc_frequency[doc.metadata["source"]] = 0
        doc_frequency[doc.metadata["source"]] += 1  # Increment frequency for retrieved documents

    return documents,action,probs, vals



# Check if the agent exists in session state
if "agent" not in st.session_state:
    # Main agent and database initialization
    n_actions = 5
    input_dims = 768  # Output size of sentence-transformers/all-mpnet-base-v2
    st.session_state.agent = Agent(n_actions=n_actions, input_dims=input_dims, epsilon=0.2)
    st.session_state.db = load_vector_store()  # Load FAISS vector store
    st.session_state.score_history = []
    st.session_state.learn_iters = 0
    st.session_state.avg_score = 0
    st.session_state.n_steps = 0
    st.session_state.best_score = -float("inf")  # Initial best score
    st.session_state.score = 0  # Reset score for each iteration
    st.session_state.episode_done = False  # Flag to track if episode is done

# Assign variables from session state
agent = st.session_state.agent
db = st.session_state.db
MAX_ATTEMPTS = 8

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User query input
user_query = st.chat_input("Your message", key=f"chat_input_{st.session_state.query_count}")

if user_query:
    # Display user's query in chat history
    with st.chat_message("human"):
        st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Generate initial AI response
    ai_response, docs, action, probs, vals = generate_answer(agent, db, query=user_query)
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    with st.chat_message("AI"):
        st.markdown(ai_response)

    # Track feedback and attempts
    feedback_value = None
    attempt_count = 0

    # Store the initial response data
    st.session_state.feedback_pending = {
        "query": user_query,
        "response": ai_response,
        "docs": [doc.metadata["source"] for doc in docs],
        "action": action,
        "probs": probs,
        "vals": vals,
    }
# Feedback collection

# Initialize attempt count in session state if not already present
if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0

# Handle feedback from the user
if st.session_state.get("feedback_pending"):
    placeholder = st.empty()
    feedback_key = f"feedback_form_{st.session_state.query_count}"  # Unique feedback key for the current query

    with placeholder.form(key=feedback_key):
        feedback = st.radio(
            "Was this response helpful?",
            options=["👍 Thumbs Up", "👎 Thumbs Down"],
            key=f"feedback_radio_{feedback_key}",
        )
        feedback_value = 1 if feedback == "👍 Thumbs Up" else -1
        comments = st.text_area(
            "Additional comments (optional):", key=f"feedback_comments_{feedback_key}"
        )
        submit_button = st.form_submit_button("Submit Feedback")
        cancel_button = st.form_submit_button("Cancel Feedback")

    if submit_button:
        # Process feedback
        pending = st.session_state.feedback_pending
        state_embedding = embedding_model.embed_query(pending["query"])

        agent.remember(
            state=state_embedding,
            action=pending["action"],
            probs=pending["probs"],
            vals=pending["vals"],
            reward=feedback_value,
            done=False,
        )

        # If negative feedback, regenerate response
        # Process feedback
        if feedback_value == -1:
            st.session_state.attempt_count += 1
            if st.session_state.attempt_count <= MAX_ATTEMPTS:  # Prevent infinite feedback loops
                # Regenerate response
                ai_response, docs, action, probs, vals = generate_answer(
                    agent, db, query=pending["query"]
                )
                st.session_state.feedback_pending = {
                    "query": pending["query"],
                    "response": ai_response,
                    "docs": [doc.metadata["source"] for doc in docs],
                    "action": action,
                    "probs": probs,
                    "vals": vals,
                }

                # Append regenerated response to chat history
                st.session_state.chat_history.append(AIMessage(content=ai_response))
                with st.chat_message("AI"):
                    st.markdown(ai_response)
            else:
                st.warning("Max attempts reached. Moving to next query.")
                st.session_state.feedback_pending = None
                st.session_state.attempt_count = 0  # Reset attempt count after exceeding max attempts

        # If positive feedback, clear pending state and move on
        else:
            st.session_state.score += feedback_value
            st.session_state.n_steps += 1
            st.session_state.feedback_pending = None
            placeholder.empty()

    if cancel_button:
        st.info("Feedback canceled.")
        st.session_state.feedback_pending = None
        placeholder.empty()

# Trigger learning every few steps
if st.session_state.n_steps % 4 == 0:
    agent.learn()
    st.session_state.learn_iters += 1